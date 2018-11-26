package simpl.interpreter;

import simpl.parser.Symbol;

public class State {

    public final Env E;
    public final Mem M;
    public final LazyTable LUT;
    public final Int p;
    

    protected State(Env E, Mem M, Int p, LazyTable lut) {
        this.E = E;
        this.M = M;
        this.p = p;
        this.LUT = lut;
    }

    public int get_pointer(){
        //garbage collection
        GC();
        //check whether there is free_pointer in M
        int pointer = this.M.get_pointer();
        //if not, allocate a new one,but not allocated in M until put!
        if(pointer == -1){            
            pointer = this.p.get();
            this.p.set(pointer+1);
        }
        return pointer;
        
    }
    
    public void GC(){
        if(gc_enable()){
            // do mark it outside because E don't has an access to memory!
            mark();
            sweep();
        }
    }
    
    public void mark(Env E){
        if(E==null)
            return;
        Symbol x = E.get_symbol();
        Value v = E.get_value();
        if(x !=null && v instanceof RefValue){
            int pointer = ((RefValue)v).p;
            this.M.mark(pointer);
        }
    }
    

    
    public boolean gc_enable(){
        return this.M.gc_enable();
    }
    
    public void sweep(){
        this.M.sweep();
    }
    
    public void mark(){
        mark(this.E);
    }

    
    public static State of(Env E, Mem M, Int p,LazyTable lut) {
        return new State(E, M, p,lut);
    }
}
