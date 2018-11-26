package simpl.parser.ast;

import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;

public class Seq extends BinaryExpr {

    public Seq(Expr l, Expr r) {
        super(l, r);
    }

    public String toString() {
        return "(" + l + " ; " + r + ")";
    }


    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult type_left_result = l.typecheck(E);
        TypeResult type_result_right = r.typecheck(E);
        Substitution s = type_result_right.s.compose(type_left_result.s);

        Type type_right = type_result_right.t;
        
        type_right = s.apply(type_right);

        return TypeResult.of(s, type_right);
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        l.eval(s);
        return r.eval(s);
    }
}
