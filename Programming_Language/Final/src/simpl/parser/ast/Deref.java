package simpl.parser.ast;

import simpl.interpreter.RefValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.typing.RefType;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;
import simpl.typing.TypeVar;

public class Deref extends UnaryExpr {

    public Deref(Expr e) {
        super(e);
    }

    public String toString() {
        return "!" + e;
    }


    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {
        TypeResult type_result = e.typecheck(E);
        Substitution s = type_result.s;
        
        Type type = type_result.t;
        type = s.apply(type);

        if(type instanceof RefType){
            return TypeResult.of(s,((RefType)type).t);
        }else if(type instanceof TypeVar){
            Type tmp = new TypeVar(false);
            s = type.unify(new RefType(tmp)).compose(s);
            tmp = s.apply(tmp);
            return TypeResult.of(s,tmp);
        }else{
            throw new TypeError("no ref type found");
        }
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        Value value = e.eval(s);
        if(!(value instanceof RefValue))
            throw new RuntimeError(" ! applied on a non-ref");
        return s.M.get( ((RefValue)value).p );
    }
}
