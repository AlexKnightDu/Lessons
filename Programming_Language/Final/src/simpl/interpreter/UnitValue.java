package simpl.interpreter;

class UnitValue extends Value {

    protected UnitValue() {
    }

    public String toString() {
        return "unit";
    }

    @Override
    public boolean equals(Object other) {
        if( other instanceof UnitValue)
            return true;
        else
            return false;
    }
}
