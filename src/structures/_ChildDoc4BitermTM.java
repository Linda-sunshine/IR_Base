package structures;

import java.util.ArrayList;

/**
 * Created by jetcai1900 on 4/27/17.
 */
public class _ChildDoc4BitermTM extends _ChildDoc {
    ArrayList<Biterm> m_bitermList;

    public _ChildDoc4BitermTM(int ID, String name, String title, String source, int ylabel) {
        super(ID, name, title, source, ylabel);
        m_bitermList = new ArrayList<Biterm>();
    }

    public void createWords4TM(int numberOfTopics, double alpha) {
        createSpace(numberOfTopics, alpha);
    }

    public void addBiterm(Biterm bitermObj){
        m_bitermList.add(bitermObj);
    }
}
