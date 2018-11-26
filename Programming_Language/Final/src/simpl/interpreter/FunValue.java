package simpl.interpreter;

import simpl.parser.Symbol;
import simpl.parser.ast.Expr;

public class FunValue extends Value {

    public final Env E;
    public final Symbol x;
    public final Expr e;

    public FunValue(Env E, Symbol x, Expr e) {
        this.E = E;
        this.x = x;
        this.e = e;
    }

    public String toString() {
        return "fun";
    }

    @Override
    public boolean equals(Object other){
        if(other instanceof FunValue){
            return x.equals(((FunValue)other).x) && e.equals(((FunValue)other).e);
        }
        return false;
    }
    
    public boolean is_rec() {
        Value value = E.get_value();
       if(value instanceof RecValue){
           //only records rec
              // if (((RecValue)value).x.equals(Symbol.symbol("f")))
                   return true;
       }
       return false;
    }
}
