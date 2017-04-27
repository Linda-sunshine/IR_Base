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

//        int wIndex = 0, wid;
//        for(_SparseFeature fv:m_x_sparse) {
//            wid = fv.getIndex();
//            for(int j=0; j<fv.getValue(); j++) {
//                m_words[wIndex] = new _Word(wid);// randomly initializing the topics inside a document
//
//                wIndex ++;
//            }
//        }
    }

    public void addBiterm(Biterm bitermObj){
        m_bitermList.add(bitermObj);
    }
}
