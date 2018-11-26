package simpl.parser.ast;

import simpl.interpreter.RefValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.typing.RefType;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;

public class Ref extends UnaryExpr {

    public Ref(Expr e) {
        super(e);
    }

    public String toString() {
        return "(ref " + e + ")";
    }

    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult typeResult = e.typecheck(E);
        Substitution s = typeResult.s;

        Type type = typeResult.t;
        type = s.apply(type);

        return TypeResult.of(s,new RefType(type));
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        int pointer = s.get_pointer();
        Value v = e.eval(s);
        //put pointer as a key for value v
        s.M.put(pointer, v);
        return new RefValue(pointer);
    }
}
