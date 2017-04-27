package structures;

/**
 * Created by jetcai1900 on 4/27/17.
 */
public class _ChildDoc4BitermTM extends _ChildDoc{
    public _ChildDoc4BitermTM(int ID, String name, String title, String source, int ylabel){
        super(ID, name, title, source, ylabel);
    }

    public void createWords4TM(int numberOfTopics, double alpha){
        createSpace(numberOfTopics, alpha);
    }
