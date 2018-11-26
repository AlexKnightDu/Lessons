package simpl.parser.ast;

import simpl.interpreter.Env;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.parser.Symbol;
import simpl.typing.Substitution;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;

public class Let extends Expr {

    public Symbol x;
    public Expr e1, e2;

    public Let(Symbol x, Expr e1, Expr e2) {
        this.x = x;
        this.e1 = e1;
        this.e2 = e2;
    }

    public String toString() {
        return "(let " + x + " = " + e1 + " in " + e2 + ")";
    }
    
    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult type_result_e1 = e1.typecheck(E);
        TypeResult type_result_e2 = e2.typecheck(TypeEnv.of(E, x, type_result_e1.t));
         Substitution s = type_result_e2.s.compose(type_result_e1.s);
        return TypeResult.of(s,type_result_e2.t);
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value_e1 = e1.eval(s);
        //put x and its value into a new Env
        State state_new = State.of(new Env(s.E, x, value_e1), s.M, s.p,s.LUT);
        return e2.eval(state_new);
    }
}
