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

public class Loop extends Expr {

    public Expr e1, e2;

    public Loop(Expr e1, Expr e2) {
        this.e1 = e1;
        this.e2 = e2;
    }

    public String toString() {
        return "(while " + e1 + " do " + e2 + ")";
    }


    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult type_result_test = e1.typecheck(E);
        TypeResult type_ersult_body = e2.typecheck(E);
        Substitution s = type_ersult_body.s.compose(type_result_test.s);

        Type type_test = type_result_test.t;
        type_test = s.apply(type_test);
        s = type_test.unify(Type.BOOL).compose(s);

        return TypeResult.of(s,Type.UNIT);
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value_e1 = e1.eval(s);
        if(!(value_e1 instanceof BoolValue))
            throw new RuntimeError("while 's test is not a value!");
        if(((BoolValue)value_e1).b){
            return new Seq(e2,this).eval(s);
        }else{
            return Value.UNIT;
        }
    }}
