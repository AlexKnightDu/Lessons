package simpl.parser.ast;

import simpl.typing.Substitution;
import simpl.typing.Type;
import simpl.typing.TypeEnv;
import simpl.typing.TypeError;
import simpl.typing.TypeResult;

public abstract class RelExpr extends BinaryExpr {

    public RelExpr(Expr l, Expr r) {
        super(l, r);
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

        s = type_right.unify(Type.INT).compose(s);
        s = type_left.unify(Type.INT).compose(s);

        return TypeResult.of(s,Type.BOOL);
    }
}
