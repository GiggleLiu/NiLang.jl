\section{Computing Fibonacci Numbers}\label{app:fib}
The following is an example that everyone likes, computing Fibonacci number recursively.

\begin{minipage}{.88\columnwidth}
    \begin{lstlisting}
using NiLang

@i function rfib(out!, n::T) where T
    n1 ← zero(T)
    n2 ← zero(T)
    @routine begin
        n1 += identity(n)
        n1 -= identity(1)
        n2 += identity(n)
        n2 -= identity(2)
    end
    if (value(n) <= 2, ~)
        out! += identity(1)
    else
        rfib(out!, n1)
        rfib(out!, n2)
    end
    ~@routine
end
\end{lstlisting}
\end{minipage}

The time complexity of this recursive algorithm is exponential to input \texttt{n}. It is also possible to write a reversible linear time with for loops.
A slightly non-trivial task is computing the first Fibonacci number that greater or equal to a certain number $z$, where a \texttt{while} statement is required.

\begin{minipage}{.88\columnwidth}
\begin{lstlisting}
@i function rfibn(n!, z)
    @safe @assert n! == 0
    out ← 0
    rfib(out, n!)
    while (out < z, n! != 0)
        ~rfib(out, n!)
        n! += identity(1)
        rfib(out, n!)
    end
    ~rfib(out, n!)
end
\end{lstlisting}
\end{minipage}

In this example, the postcondition \texttt{n!=0} in the \texttt{while} statement is false before entering the loop, and it becomes true in later iterations. In the reverse program, the \texttt{while} statement stops at \texttt{n==0}.
If executed correctly, a user will see the following result.

\begin{minipage}{.88\columnwidth}
\begin{lstlisting}
julia> rfib(0, 10)
(55, 10)

julia> rfibn(0, 100)
(12, 100)

julia> (~rfibn)(rfibn(0, 100)...)
(0, 100)
\end{lstlisting}
\end{minipage}

This example shows how an addition postcondition provided by the user can help to reverse a control flow without caching controls.

\subsection{Unitary Matrices}\label{app:umm}
A unitary matrix features uniform eigenvalues and reversibility. It is widely used as an approach to ease the gradient exploding and vanishing problem~\cite{Arjovsky2015,Wisdom2016,Li2016} and the memory wall problem~\cite{Luo2019}.
One of the simplest ways to parametrize a unitary matrix is representing a unitary matrix as a product of two-level unitary operations~\cite{Li2016}. A real unitary matrix of size $N$ can be parametrized compactly by $N(N-1)/2$ rotation operations~\cite{Li2013}
\begin{align}
    {\rm ROT}(a!, b!, \theta)  = \left(\begin{matrix}
        \cos(\theta) & - \sin(\theta)\\
        \sin(\theta)  & \cos(\theta)
    \end{matrix}\right)
    \left(\begin{matrix}
        a!\\
        b!
    \end{matrix}\right),
\end{align}
where \texttt{$\theta$} is the rotation angle, \texttt{a!} and \texttt{b!} are target registers.

\begin{minipage}{.88\columnwidth}
\begin{lstlisting}[mathescape=true]
using NiLang, NiLang.AD

@i function umm!(x!, θ)
    @safe @assert length(θ) == 
            length(x!)*(length(x!)-1)/2
    k ← 0
    for j=1:length(x!)
        for i=length(x!)-1:-1:j
            k += identity(1)
            ROT(x![i], x![i+1], θ[k])
        end
    end

    k → length(θ)
end
\end{lstlisting}
\end{minipage}

Here, the ancilla \texttt{k} is deallocated manually by specifying its value, because we know the loop size is $N(N-1)/2$.
We define the test functions in order to check gradients.

\begin{minipage}{\columnwidth}
\begin{lstlisting}[mathescape=true,multicols=2]
julia> @i function test!(out!, x!::Vector, θ::Vector)
           umm!(x!, θ)
           isum(out!, x!)
       end

julia> out, x, θ = 0.0, randn(4), randn(6);

julia> @instr Grad(test!)(Val(1), out, x, θ)

julia> x
4-element Array{GVar{Float64,Float64},1}:
 GVar(1.220182125326287, 0.14540743042341095) 
 GVar(2.1288634811475937, -1.3749962375499805)
 GVar(1.2696579252569677, 1.42868739498625)   
 GVar(0.1083891125379283, 0.2170123344615735) 

julia> @instr (~test!')(Val(1), out, x, θ)

julia> x
4-element Array{Float64,1}:
 1.220182125326287  
 2.1288634811475933 
 1.2696579252569677 
 0.10838911253792821
\end{lstlisting}
\end{minipage}

In the above testing code, \texttt{Grad(test)} attaches a gradient field to each element of \texttt{x}. \texttt{$\sim$Grad(test)} is the inverse program that erase the gradient fields.
Notably, this reversible implementation costs zero memory allocation, although it changes the target variables inplace.

%It implements the backward rule of a \texttt{ROT} instruction
%\begin{align}
%    \begin{split}
%    \overline{\theta}  &= \sum\frac{\partial R(\theta)}{\partial \theta}\odot(\overline{out!}x^T)\\
%    &= \Tr\left[\frac{\partial R(\theta)}{\partial \theta}^T\overline{out!}x^T\right]\\
%    &= \Tr\left[R\left(\frac{\pi}{2}-\theta\right)\overline{out!}x^T\right]
%    \end{split}
%\end{align}
