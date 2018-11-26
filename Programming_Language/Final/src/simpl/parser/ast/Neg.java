package simpl.parser.ast;

import simpl.interpreter.IntValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;

public class Neg extends UnaryExpr {

    public Neg(Expr e) {
        super(e);
    }

    public String toString() {
        return "~" + e;
    }


    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult typeResult = e.typecheck(E);
        Substitution s = typeResult.s;

        Type type = typeResult.t;
        type = s.apply(type);

        s = type.unify(Type.INT).compose(s);
        return TypeResult.of(s,Type.INT);
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value = e.eval(s);
        if(!(value instanceof IntValue)){
            throw new RuntimeError("neg applied on a non-int!");
        }
        return new IntValue(0-((IntValue)value).n);
    }
}
