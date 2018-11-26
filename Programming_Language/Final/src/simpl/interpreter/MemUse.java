package simpl.interpreter;

public class MemUse{
    public Value value;
    public boolean mark;
    public MemUse(Value value) {
        this.value = value;
        mark = false;
    }
}
