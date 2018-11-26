package simpl.parser.ast;

import simpl.interpreter.PairValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.typing.PairType;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;

public class Pair extends BinaryExpr {

    public Pair(Expr l, Expr r) {
        super(l, r);
    }

    public String toString() {
        return "(pair " + l + " " + r + ")";
    }


    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult type_result_left = l.typecheck(E);
        TypeResult type_result_right = r.typecheck(E);

        Substitution s = type_result_right.s.compose(type_result_left.s);

        Type type_left = s.apply(type_result_left.t);
        Type type_right = s.apply(type_result_right.t);

        return TypeResult.of(s,new PairType(type_left,type_right));
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value_left = l.eval(s);
        Value value_right = r.eval(s);
        return new PairValue(value_left, value_right);
    }
}
