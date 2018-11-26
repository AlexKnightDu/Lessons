package simpl.parser.ast;

import simpl.interpreter.Env;
import simpl.interpreter.RecValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.parser.Symbol;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;
import simpl.typing.TypeVar;

public class Rec extends Expr {

    public Symbol x;
    public Expr e;

    public Rec(Symbol x, Expr e) {
        this.x = x;
        this.e = e;
    }

    public String toString() {
        return "(rec " + x + "." + e + ")";
    }

    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeVar a = new TypeVar(false);
        TypeResult type_result_body = e.typecheck(TypeEnv.of(E,x,a));

        Substitution s = type_result_body.s;
        Type tmp = type_result_body.t;

        tmp = s.apply(tmp);
        s = tmp.unify(s.apply(a)).compose(s);

        return TypeResult.of(s,s.apply(tmp));
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        State state_new = State.of(new Env(s.E, x, new RecValue(s.E, x, e)), s.M, s.p,s.LUT);
        return e.eval(state_new);
    }
}
