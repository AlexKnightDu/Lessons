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
import simpl.typing.TypeVar;

public class Assign extends BinaryExpr {

    public Assign(Expr l, Expr r) {
        super(l, r);
    }

    public String toString() {
        return l + " := " + r;
    }


    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {

        TypeResult type_result_left = l.typecheck(E);
        TypeResult type_result_right = r.typecheck(E);
        Substitution s = type_result_right.s.compose(type_result_left.s);

        Type type_left = type_result_left.t;
        Type type_right = type_result_right.t;

        type_left = s.apply(type_left);
        type_right = s.apply(type_right);

        Type tmp = new TypeVar(false);
        s = type_left.unify(new RefType(tmp)).compose(s);
        tmp = s.apply(tmp);
        s = type_right.unify(tmp).compose(s);

        return TypeResult.of(s,Type.UNIT);
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value_left = l.eval(s);
        if(!(value_left instanceof RefValue))
            throw new RuntimeError("cannot assign to a nonref variable!");
        Value value_right = r.eval(s);
        // put left value's pointer and its value into Memory!
        s.M.put(((RefValue) value_left).p, value_right);
        return Value.UNIT;
    }
}
