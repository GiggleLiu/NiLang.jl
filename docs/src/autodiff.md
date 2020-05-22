%One can check the correctness of this definition as follows.

%\begin{minipage}{.88\columnwidth}
%\begin{lstlisting}[mathescape=true]
%julia> using NiLang, NiLang.AD

%julia> a, b, y = GVar(0.5), GVar(0.6), GVar(0.9)
%(GVar(0.5, 0.0), GVar(0.6, 0.0), GVar(0.9, 0.0))

%julia> @instr grad(y) += identity(1.0)

%julia> @instr y += a * b
%GVar(0.6, -0.5)

%julia> a, b, y
%(GVar(0.5, -0.6), GVar(0.6, -0.5), GVar(1.2, 1.0))

%julia> @instr y -= a * b
%GVar(0.6, 0.0)

%julia> a, b, y
%(GVar(0.5, 0.0), GVar(0.6, 0.0), GVar(0.899999, 1.0))
%\end{lstlisting}
%\end{minipage}

%Here, since \texttt{J((:-=)(*)) = J((:+=)(*))${}^{-1}$}, consecutively applying them will restore the gradient fields of all variables.
%More local Jacobians and Hessians for basic instructions used in this section could be found in \App{app:jacobians}.


