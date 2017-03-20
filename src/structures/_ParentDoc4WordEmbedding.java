package structures;

import java.util.HashMap;

/**
 * Created by jetcai1900 on 3/18/17.
 *
 * add similarity of each word in the article to all words in comments within the thread
 *
 */
public class _ParentDoc4WordEmbedding extends _ParentDoc {
    public double[][][] m_commentThread_wordSS;

    public _ParentDoc4WordEmbedding(int ID, String name, String title, String source, int ylabel){
        super(ID, name, title, source, ylabel);
    }

    public void setTopics4Gibbs(int k, double alpha, int vocSize, int gammaSize) {
        createSpace(k, alpha);

        m_commentThread_wordSS = new double[gammaSize][k][vocSize];

        int wIndex = 0, wid, tid;
        for (_SparseFeature fv : m_x_sparse) {
            wid = fv.getIndex();
            for (int j = 0; j < fv.getValue(); j++) {
                tid = m_rand.nextInt(k);
                m_words[wIndex] = new _Word(wid, tid);// randomly initializing the topics inside a document
                m_sstat[tid]++; // collect the topic proportion

                wIndex++;
            }
        }

        m_phi = new double[m_x_sparse.length][k];
        m_word2Index = new HashMap<Integer, Integer>();
        for (int i = 0; i < m_x_sparse.length; i++)
            m_word2Index.put(m_x_sparse[i].m_index, i);
    }


}
