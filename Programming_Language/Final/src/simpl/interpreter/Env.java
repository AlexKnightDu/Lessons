package simpl.interpreter;

import simpl.parser.Symbol;

public class Env {

    private final Env E;
    private final Symbol x;
    private final Value v;

    private Env() {
        E = null;
        x = null;
        v = null;
    }

    public static Env empty = new Env() {
        public Value get(Symbol y) {
            return null;
        }

        public Env clone() {
            return this;
        }
    };

    public Env(Env E, Symbol x, Value v) {
        this.E = E;
        this.x = x;
        this.v = v;
    }

    public Value get(Symbol y) {
        // if y==x, return v, else find the value in E
        if(y.toString().equals(x.toString())){
            return v;
        }
        return E.get(y);
    }

    public Env clone() {
        return new Env(E,x,v);
    }
    
    public String toString(){
        String str = x.toString()+":"+v.toString()+"\n";
        str += E.toString();
        return str;
    }
    
    public Symbol get_symbol(){
        return x;
    }
    
    public Value get_value(){
        return v;
    }
    
    public Env get_env(){
        return E;
    }    
}
