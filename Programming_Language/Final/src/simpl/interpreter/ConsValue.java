package simpl.interpreter;

public class ConsValue extends Value {

    public final Value v1, v2;
    public int length = 0;
    
    public ConsValue(Value v1, Value v2) {
        this.v1 = v1;
        this.v2 = v2;
        int length1 = 0;
        int length2=0;
        
        if(v1 instanceof ConsValue)
            length1 = ((ConsValue)v1).length;
        else if(v1 instanceof NilValue)
            length1 = 0;
        else {
            length1= 1;
        }
        
        if(v2 instanceof ConsValue)
            length2 = ((ConsValue)v2).length;
        else if(v2 instanceof NilValue)
            length2 = 0;
        else {
            length2= 1;
        }
        
        this.length  = length1+length2;
    }

    public String toString() {
        return "list@"+length;
    }

    @Override
    public boolean equals(Object other) {
        if(other instanceof NilValue){
            return false;
        }else if(other instanceof ConsValue){
            return v1.equals(((ConsValue) other).v1) && v2.equals(((ConsValue) other).v2);
        }
        return false;
    }
    
}
