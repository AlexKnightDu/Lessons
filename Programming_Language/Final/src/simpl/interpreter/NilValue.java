package simpl.interpreter;

class NilValue extends Value {

    protected NilValue() {
    }

    public String toString() {
        return "nil";
    }

    @Override
    public boolean equals(Object other) {
        //need to take care of ConsValue here,otherwise an ConsValue will be taken as a NIL 
        if(other instanceof ConsValue){
            return false;
        }else if(other instanceof NilValue){
            return true;
        }
        return false;
    }
}
