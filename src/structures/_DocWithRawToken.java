package structures;

import java.util.ArrayList;

/**
 * Created by jetcai1900 on 4/27/17.
 */
public class _DocWithRawToken extends _Doc {
    ArrayList<Biterm> m_bitermList;

    public _DocWithRawToken(int ID, String source, int ylabel){
        super(ID, source, ylabel);
        m_bitermList = new ArrayList<Biterm>();
    }

    public void createWords4TM(int numberOfTopics, double alpha){
        createSpace(numberOfTopics, alpha);

    }

    public void addBiterm(Biterm bitermObj){
        m_bitermList.add(bitermObj);
    }
}
