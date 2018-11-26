package simpl.parser.ast;

import simpl.interpreter.Env;
import simpl.interpreter.FunValue;
import simpl.interpreter.RuntimeError;
import simpl.interpreter.State;
import simpl.interpreter.Value;
import simpl.interpreter.FuncEntry;
import simpl.parser.Symbol;
import simpl.typing.ArrowType;
import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;
import simpl.typing.TypeVar;

public class App extends BinaryExpr {

    public App(Expr l, Expr r) {
        super(l, r);
    }

    public String toString() {
        return "(" + l + " " + r + ")";
    }

    @Override
    public TypeResult typecheck(TypeEnv E) throws TypeError {

        TypeResult type_result_left = l.typecheck(E);
        TypeResult type_result_right = r.typecheck(E);
        Substitution s = type_result_right.s.compose(type_result_left.s);

        Type type_left = type_result_left.t;
        Type type_right = type_result_right.t;

        type_left = s.apply(type_left);
        type_right = s.apply(type_right);

        if(type_left instanceof ArrowType) {
            s = ((ArrowType) type_left).t1.unify(type_right).compose(s);//type_left=type_right->a
            type_left = s.apply(type_left);
            return TypeResult.of(s, ((ArrowType) type_left).t2);
        }else if(type_left instanceof TypeVar){
            TypeVar tmp = new TypeVar(false);//new type a
            s = type_left.unify(new ArrowType(type_right,tmp)).compose(s);
             return TypeResult.of(s, s.apply(tmp));
        }else{
            throw new TypeError("no function found");
        }
    }

    @Override
    public Value eval(State s) throws RuntimeError {
        
        Value value_e1 = l.eval(s);
        if( !(value_e1 instanceof FunValue) )
            throw new RuntimeError("not a function");
        // v is a function
        FunValue fun = (FunValue) value_e1;
        // evaluate e2
        Value value_e2 = r.eval(s);
        // put parameter's  value into Env
        
        if(fun.is_rec()){
            FuncEntry fe = new FuncEntry(value_e1,value_e2);
                    
            Value result = s.LUT.get_result(fe);
         
            if(result ==  null ){
                State state_new = State.of( new Env(fun.E,fun.x,value_e2), s.M , s.p,s.LUT);
                //evaluate function body's value
                result =  fun.e.eval(state_new);
                s.LUT.put(fe,result);
            }
            return result;
    }
    else{
        State state_new = State.of( new Env(fun.E,fun.x,value_e2), s.M , s.p,s.LUT);
        //evaluate function body's value
        return  fun.e.eval(state_new);    
    }
  }
}
