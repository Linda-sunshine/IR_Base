package topicmodels.correspondenceModels;

import structures.*;

import java.util.Arrays;
import java.util.Collection;

/**
 * Created by jetcai1900 on 4/27/17.
 */
public class BitermTopicModel4AC extends BitermTopicModel{
    public BitermTopicModel4AC(int number_of_iteration, double converge, double beta,
                               _Corpus c, double lambda,
                               int number_of_topics, double alpha, double burnIn, int lag){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);
    }

    @Override
    public String toString() {
        return String.format("BitermTopicModel4AC[k:%d, alpha:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta, 1-m_testWord4PerplexityProportion);
    }

    @Override
    protected void initialize_probability(Collection<_Doc> collection) {
        for(int i=0; i<number_of_topics; i++)
            Arrays.fill(word_topic_sstat[i], d_beta);
        Arrays.fill(m_sstat, d_beta*vocabulary_size);
        Arrays.fill(m_topicProbCache, 0);
        Arrays.fill(m_bitermTopicStat, 0);
        Arrays.fill(m_topics, 0);

        double avgDocLen = 0;
        // initialize topic-word allocation, p(w|z)
        for(_Doc d:collection) {
            avgDocLen += d.getTotalDocLength();
            constructBiterms(d);
        }

        for(Biterm bitermObj : m_bitermList){
            m_bitermTopicStat[bitermObj.getTopic()] ++;
            for(_Word w:bitermObj.getWords()){
                word_topic_sstat[w.getTopic()][w.getIndex()] ++;
                m_sstat[w.getTopic()] ++;

            }
        }

        avgDocLen = avgDocLen*1.0/collection.size();
        System.out.println("avg len\t"+avgDocLen);
        imposePrior();
    }

    //construct biterms
    protected void constructBiterms(_Doc d){
        if(d instanceof  _ParentDoc) {
            return;
        }
//        _DocWithRawToken doc = (_DocWithRawToken)d;
        _ChildDoc cDoc = (_ChildDoc)d;

        // whether we can move this to initialization
//        cDoc.createWords4TM(number_of_topics, d_alpha);
        cDoc.createSpace(number_of_topics, d_alpha);

        for(int i=0; i<d.getTotalDocLength(); i++){
            for(int j=i+1; j<d.getTotalDocLength(); j++){
                int wid1 = d.getWordByIndex(i).getIndex();
                int wid2 = d.getWordByIndex(j).getIndex();
                Biterm bitermObj = new Biterm(wid1, wid2);
                bitermObj.setTopics4GibbsbyRawToken(number_of_topics);
                m_bitermList.add(bitermObj);
                .addBiterm(bitermObj);
            }
        }

    }
}
