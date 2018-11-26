package simpl.parser.ast;

import simpl.interpreter.BoolValue;
import simpl.interpreter.IntValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;

public class GreaterEq extends RelExpr {

    public GreaterEq(Expr l, Expr r) {
        super(l, r);
    }

    public String toString() {
        return "(" + l + " >= " + r + ")";
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value_left = l.eval(s);
        Value value_right = r.eval(s);
        if( !(value_left instanceof IntValue) ) 
            throw new RuntimeError("left hand  isn't int values");
        if(  !(value_right instanceof IntValue) )
            throw new RuntimeError("right hand  isn't int values");
        return new BoolValue( ((IntValue)value_left).n >= ((IntValue)value_right).n );
    }
}
