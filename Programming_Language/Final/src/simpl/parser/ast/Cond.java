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
import simpl.typing.TypeVar;

public class Cond extends Expr {

    public Expr e1, e2, e3;

    public Cond(Expr e1, Expr e2, Expr e3) {
        this.e1 = e1;
        this.e2 = e2;
        this.e3 = e3;
    }

    public String toString() {
        return "(if " + e1 + " then " + e2 + " else " + e3 + ")";
    }


    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult type_result_test = e1.typecheck(E);
        TypeResult type_result_then = e2.typecheck(E);
        TypeResult type_result_else = e3.typecheck(E);

        TypeVar resultType = new TypeVar(false);

        Type type_test = type_result_test.t;
        Type type_then = type_result_then.t;
        Type type_else = type_result_else.t;

        Substitution s = type_result_else.s.compose(type_result_then.s).compose(type_result_test.s);

        type_test = s.apply(type_test);
        type_then = s.apply(type_then);
        type_else = s.apply(type_else);

        s = type_test.unify(Type.BOOL).compose(s);
        type_then = s.apply(type_then);
        s = type_then.unify(resultType).compose(s);
        type_else = s.apply(type_else);
        s = type_else.unify(resultType).compose(s);

        return TypeResult.of(s,s.apply(resultType));
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value_left = e1.eval(s);
        if(!(value_left instanceof BoolValue)
            )throw new RuntimeError("if's test is not a boolean");
        if(((BoolValue)value_left).b){
            return e2.eval(s);
        }else{
            return e3.eval(s);
        }
    }
}
