package structures;

/**
 * Created by jetcai1900 on 3/18/17.
 *
 * add x_sstat, x_proportion
 */
public class _ChildDoc4WordEmbedding extends _ChildDoc{

    double x_proportion; //the ratio x=1
    int x_sstat; // the number of words assigned to x=1

    public _ChildDoc4WordEmbedding(int ID, String name, String title, String source, int ylabel){
        super(ID, name, title, source, ylabel);
    }


    ////initialize x, compute the x_ratio, x_ss



}
