package simpl.parser.ast;

import simpl.interpreter.FunValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.parser.Symbol;
import simpl.typing.ArrowType;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;
import simpl.typing.TypeVar;

public class Fn extends Expr {

    public Symbol x;
    public Expr e;

    public Fn(Symbol x, Expr e) {
        this.x = x;
        this.e = e;
    }

    public String toString() {
        return "(fn " + x + "." + e + ")";
    }

    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        Type a = new TypeVar(false);//x: a
        TypeResult type_result_body = e.typecheck(TypeEnv.of(E, x, a));//u==>e:t

        Type b = new TypeVar(false);//f: a->b

        Substitution s = type_result_body.t.unify(b).compose(type_result_body.s);
        a = s.apply(a);
        b = s.apply(b);

        return TypeResult.of(s,new ArrowType(a,b));
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        return new FunValue(s.E,x,e);
    }
}
