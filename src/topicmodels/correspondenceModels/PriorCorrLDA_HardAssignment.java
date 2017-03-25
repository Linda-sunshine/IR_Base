package topicmodels.correspondenceModels;

import structures.*;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by jetcai1900 on 3/24/17.
 */
public class PriorCorrLDA_HardAssignment extends PriorCorrLDA{
    public PriorCorrLDA_HardAssignment(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
                                       int number_of_topics, double alpha, double alpha_c, double burnIn, int lag){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics,  alpha, alpha_c, burnIn, lag);
    }

    protected void sampleInChildDoc(_Doc d) {
        _ChildDoc cDoc = (_ChildDoc) d;
        int wid, tid;
        double normalizedProb = 0;
        _ParentDoc pDoc = (_ParentDoc)(cDoc.m_parentDoc);

        HashMap<Integer, ArrayList<Integer>> wordTopicMap = new HashMap<Integer, ArrayList<Integer>>();
        for(_Word w:pDoc.getWords()){
            wid = w.getIndex();
            tid = w.getTopic();

            if(wordTopicMap.containsKey(wid)){
                ArrayList<Integer> topicList = wordTopicMap.get(wid);
                topicList.add(tid);
            }else{
                ArrayList<Integer> topicList = new ArrayList<Integer>();
                topicList.add(tid);
                wordTopicMap.put(wid, topicList);
            }

        }

        for (_Word w : cDoc.getWords()) {
            wid = w.getIndex();
            tid = w.getTopic();

            cDoc.m_sstat[tid]--;
            if(m_collectCorpusStats){
                word_topic_sstat[tid][wid] --;
                m_sstat[tid] --;
            }

            if(wordTopicMap.containsKey(wid)) {
                for (int k : wordTopicMap.get(wid)) {
                    m_topicProbCache[k] += 1;
                }

                for(int k=0; k<number_of_topics; k++) {
                    m_topicProbCache[k] /= wordTopicMap.get(wid).size();
                    normalizedProb += m_topicProbCache[k];
                }

            }else {

                normalizedProb = 0;
                for (tid = 0; tid < number_of_topics; tid++) {
                    double pWordTopic = childWordByTopicProb(tid, wid);
                    double pTopicDoc = childTopicInDocProb(tid, cDoc);

                    m_topicProbCache[tid] = pWordTopic * pTopicDoc;
                    normalizedProb += m_topicProbCache[tid];
                }
            }

            normalizedProb *= m_rand.nextDouble();
            for (tid = 0; tid < number_of_topics; tid++) {
                normalizedProb -= m_topicProbCache[tid];
                if (normalizedProb < 0)
                    break;
            }

            if (tid == number_of_topics)
                tid--;

            w.setTopic(tid);
            cDoc.m_sstat[tid]++;
            if(m_collectCorpusStats){
                word_topic_sstat[tid][wid] ++;
                m_sstat[tid] ++;
            }
        }
    }
}
