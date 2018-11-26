package simpl.interpreter;

import java.util.Stack;

public class LazyTable {
    private Stack<FuncEntry> table ;
        
        public LazyTable() {
            table = new Stack<FuncEntry>();
        }
               
        public Value get_result(FuncEntry fe){
            for (FuncEntry f:table){
                if (f.equal(fe)){
                        System.out.println("Reuse stored evaluations: "+ fe.fun + " " +fe.para);
                        return f.result;
                }
            }
            return null;
        }
        
        public void put(FuncEntry fe,Value result) {
            fe.set_result(result);
            table.push(fe);
            clear();
        }
        
        public void clear(){
            if(table.size()>200){
                while(table.size()>200){
                    table.pop();
                }
            }
        }
        
        
}
