package simpl.parser.ast;

import simpl.interpreter.ConsValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.typing.ListType;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;

public class Cons extends BinaryExpr {

    public Cons(Expr l, Expr r) {
        super(l, r);
    }

    public String toString() {
        return "(" + l + " :: " + r + ")";
    }

    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult type_result_left = l.typecheck(E);
        TypeResult type_result_right = r.typecheck(E);
        Substitution s = type_result_right.s.compose(type_result_left.s);
        //cannot be written as below ! Since we first evaluate left , so we have to use left hand's info first.
        //Substitution s = type_result_left.s.compose(type_result_right.s);
        Type type_left = type_result_left.t;
        Type type_right = type_result_right.t;

        type_left = s.apply(type_left);
        type_right = s.apply(type_right);

        s = type_right.unify(new ListType(type_left)).compose(s);

        return TypeResult.of(s,s.apply(type_right));
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value_left = l.eval(s);
        Value value_right = r.eval(s);
        return new ConsValue(value_left, value_right);
    }
}
