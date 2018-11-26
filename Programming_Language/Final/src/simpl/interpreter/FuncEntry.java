package simpl.interpreter;

public class FuncEntry {

        Value fun;
        Value para;
        Value result;
        
        public FuncEntry(Value expr,Value p) {
            fun = expr;
            para = p;
        }
        
        public boolean equal(FuncEntry f){
            return para.equals(f.para)&& fun.equals(f.fun);        
        }
        
        public void set_result(Value r){
            this.result = r;
        }
}
