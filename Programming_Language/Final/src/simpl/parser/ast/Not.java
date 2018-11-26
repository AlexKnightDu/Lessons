package simpl.parser.ast;

import simpl.interpreter.BoolValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;

public class Not extends UnaryExpr {

    public Not(Expr e) {
        super(e);
    }

    public String toString() {
        return "(not " + e + ")";
    }


    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult typeResult = e.typecheck(E);
        Substitution s = typeResult.s;
        
        Type type = typeResult.t;
        type = s.apply(type);

        s = type.unify(Type.BOOL).compose(s);
        return TypeResult.of(s,Type.BOOL);
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value = e.eval(s);
        if(!(value instanceof BoolValue))
            throw new RuntimeError("not applied on a non-boolean!");
        return new BoolValue(!(((BoolValue)value).b));
    }
}
