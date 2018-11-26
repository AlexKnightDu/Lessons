package simpl.typing;

import simpl.parser.Symbol;

public class DefaultTypeEnv extends TypeEnv {

    private TypeEnv E;

    public DefaultTypeEnv() {
        Type t1 = new TypeVar(true);
        Type t2 = new TypeVar(true);
        E = TypeEnv.empty;
        E = TypeEnv.of(E,Symbol.symbol("NUM") , Type.INT);
        E = TypeEnv.of(E, Symbol.symbol("TRUE") , Type.BOOL);
        E = TypeEnv.of(E, Symbol.symbol("FALSE") , Type.BOOL);
        E = TypeEnv.of(E, Symbol.symbol("UNIT") , Type.UNIT);
        E = TypeEnv.of(E, Symbol.symbol("iszero"), new ArrowType(Type.INT, Type.BOOL));
        E = TypeEnv.of(E, Symbol.symbol("pred"), new ArrowType(Type.INT, Type.INT));
        E = TypeEnv.of(E, Symbol.symbol("succ"), new ArrowType(Type.INT, Type.INT));
        E = TypeEnv.of(E, Symbol.symbol("fst"), new ArrowType(new PairType(t1,t2), t1));
        E = TypeEnv.of(E, Symbol.symbol("snd"), new ArrowType(new PairType(t1,t2), t2));
        E = TypeEnv.of(E, Symbol.symbol("hd"), new ArrowType(new ListType(t1), t1));
        E = TypeEnv.of(E, Symbol.symbol("tl"), new ArrowType(new ListType(t1), new ListType(t1)));
    }

    @Override
    public Type get(Symbol x) {
        return E.get(x);
    }
}
