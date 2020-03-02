# NiLang Grammar

To define a reversible function one can use macro **@i** plus a function definition like bellow

```julia
"""
docstring...
"""
@i function f(args..., kwargs...) where {...}
    <stmts>
end
```

where the definition of **<stmts>** are shown in the grammar bellow.
The following is a list of terminologies used in the definition of grammar

* <ident>, symbols
* <num>, numbers
* $0$, empty statement
* <JuliaExpr>, native Julia expression
* $[$ $]$,  zero or one repetitions.

Here, all $JuliaExpr$ should be pure, otherwise the reversibility is not guaranteed.
Dataview is a view of a data, it can be a bijective mapping of an object, an item of an array or a field of an object.


```bnf
Stmts : 0 
      | Stmt
      | Stmts Stmt
      ;

Stmt : BlockStmt
     | IfStmt
     | WhileStmt
     | ForStmt
     | InstrStmt
     | RevStmt
     | AncillaStmt
     | TypecastStmt 
     | @routine Stmt
     | @safe <JuliaExpr>
     | CallStmt
     ;


BlockStmt : 'begin' Stmts 'end';

RevCond : '(' <JuliaExpr> ',' <JuliaExpr> ')';

IfStmt : 'if' RevCond Stmts ['else' Stmts] 'end';

WhileStmt : 'while' RevCond Stmts 'end';

Range : <JuliaExpr> ':' <JuliaExpr> [':' <JuliaExpr>];

ForStmt : 'for' <ident> '=' Range Stmts 'end';

KwArg : <ident> '=' <JuliaExpr>;

KwArgs : [KwArgs ','] KwArg ;

CallStmt : <JuliaExpr> '(' [DataViews] [';' KwArgs] ')';

Constant : <num> | 'π';

InstrBinOp : '+=' | '-=' | '⊻=';

InstrTrailer : ['.'] '(' [DataViews] ')';

InstrStmt : DataView InstrBinOp <ident> [InstrTrailer];

RevStmt : '~' Stmt;

AncillaStmt : <ident> '←' <JuliaExpr>
            | <ident> '→' <JuliaExpr>
            ;

TypecastStmt : '(' <JuliaExpr> '=>' <JuliaExpr> ')' '(' <ident> ')';

@routine : '@routine' <ident> Stmt;

@safe : '@safe' <JuliaExpr>;

DataViews : 0
          | DataView
          | DataViews ',' DataView
          | DataViews ',' DataView '...'
          ;

DataView : DataView '[' <JuliaExpr> ']'
         | DataView '.' <ident>
         | <JuliaExpr> '(' DataView ')'
         | DataView '\''
         | '-' DataView
         | Constant
         | <ident>
         ;
```
