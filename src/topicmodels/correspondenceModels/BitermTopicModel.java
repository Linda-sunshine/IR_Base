package topicmodels.correspondenceModels;

import structures.*;
import topicmodels.LDA.LDA_Gibbs;
import utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

/**
 * Created by jetcai1900 on 4/27/17.
 */
public class BitermTopicModel extends LDA_Gibbs {
    ArrayList<Biterm> m_bitermList;
    double[] m_bitermTopicStat;
    double[] m_topics;

    public BitermTopicModel(int number_of_iteration, double converge, double beta,
                            _Corpus c, double lambda,
                            int number_of_topics, double alpha, double burnIn, int lag){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag);
        m_bitermList = new ArrayList<Biterm>();
        m_bitermTopicStat = new double[number_of_topics];
        m_topics = new double[number_of_topics];
    }

    @Override
    public String toString() {
        return String.format("Biterm topic model[k:%d, alpha:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta, 1-m_testWord4PerplexityProportion);
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

        System.out.println("biterm size\t"+m_bitermList.size());
        avgDocLen = avgDocLen*1.0/collection.size();
        System.out.println("avg len\t"+avgDocLen);
        imposePrior();
    }

    //construct biterms
    protected void constructBiterms(_Doc d){
        _DocWithRawToken doc = (_DocWithRawToken)d;

        // whether we can move this to initialization
        doc.createWords4TM(number_of_topics, d_alpha);
        for(int i=0; i<d.getTotalDocLength(); i++){
            for(int j=i+1; j<d.getTotalDocLength(); j++){
                int wid1 = d.getWordByIndex(i).getIndex();
                int wid2 = d.getWordByIndex(j).getIndex();
                Biterm bitermObj = new Biterm(wid1, wid2);
                bitermObj.setTopics4GibbsbyRawToken(number_of_topics);
                m_bitermList.add(bitermObj);
                doc.addBiterm(bitermObj);
            }
        }

    }

    @Override
    protected void init() {
        //we just simply permute the training instances here
        int t;
        Biterm tmpBiterm;
        for(int i=m_bitermList.size()-1; i>1; i--) {
            t = m_rand.nextInt(i);

            tmpBiterm = m_bitermList.get(i);
            m_bitermList.set(i, m_bitermList.get(t));
            m_bitermList.set(t, tmpBiterm);

        }
    }

    public void EM() {
        System.out.format("Starting %s...\n", toString());

        long starttime = System.currentTimeMillis();

        m_collectCorpusStats = true;
        initialize_probability(m_trainSet);

//		double delta, last = calculate_log_likelihood(), current;
        double delta=0, last=0, current=0;
        int i = 0, displayCount = 0;
        do {
            init();

            calculate_E_step();

            calculate_M_step(i);

            if (m_converge>0 || (m_displayLap>0 && i%m_displayLap==0 && displayCount > 6)){//required to display log-likelihood
                current = calculate_log_likelihood();//together with corpus-level log-likelihood
//				current += calculate_log_likelihood();//together with corpus-level log-likelihood

                if (i>0)
                    delta = (last-current)/last;
                else
                    delta = 1.0;
                last = current;
            }

            if (m_displayLap>0 && i%m_displayLap==0) {
                if (m_converge>0) {
                    System.out.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
                    infoWriter.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);

                } else {
                    System.out.print(".");
                    if (displayCount > 6){
                        System.out.format("\t%d:%.3f\n", i, current);
                        infoWriter.format("\t%d:%.3f\n", i, current);
                    }
                    displayCount ++;
                }
            }

            if (m_converge>0 && Math.abs(delta)<m_converge)
                break;//to speed-up, we don't need to compute likelihood in many cases
        } while (++i<this.number_of_iteration);

        finalEst();

        long endtime = System.currentTimeMillis() - starttime;
        System.out.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);
        infoWriter.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);

    }

    public double calculate_E_step() {
        double p;
        int wid, tid;

        for(Biterm bitermObj:m_bitermList){
            tid = bitermObj.getTopic();
            m_bitermTopicStat[tid] --;

            for(_Word w:bitermObj.getWords()) {
                wid = w.getIndex();
                if (m_collectCorpusStats) {
                    word_topic_sstat[tid][wid]--;
                    m_sstat[tid]--;
                }
            }

            p = 0;
            for(tid=0; tid<number_of_topics; tid++){
                m_topicProbCache[tid] = topicInDocProb(tid)*wordByTopicProb(tid, bitermObj);
                p += m_topicProbCache[tid];
            }

            p *= m_rand.nextDouble();

            for(tid=0; tid<number_of_topics; tid++){
                p -= m_topicProbCache[tid];
                if(p<0)
                    break;
            }

            if(tid==number_of_topics)
                tid --;

            //assign the selected topic to word
            bitermObj.setTopic(tid);
            m_bitermTopicStat[tid] ++;
            for(_Word w:bitermObj.getWords()) {
                wid = w.getIndex();
                w.setTopic(tid);
                if (m_collectCorpusStats) {
                    word_topic_sstat[tid][wid]++;
                    m_sstat[tid]++;
                }
            }
        }
        return 0;
    }

    public double topicInDocProb(int tid){
        double prob = m_bitermTopicStat[tid] + d_alpha;
        return prob;
    }

    public double wordByTopicProb(int tid, Biterm bitermObj){
        double prob = 1;
        int wid = 0;

        _Word w1 = bitermObj.getWords()[0];
        _Word w2 = bitermObj.getWords()[1];

        prob *= word_topic_sstat[tid][w1.getIndex()]/(m_sstat[tid]+1);
        prob *= word_topic_sstat[tid][w2.getIndex()]/(m_sstat[tid]);

        return prob;
    }

    public void calculate_M_step(int iter) {
        //literally we do not have M-step in Gibbs sampling
        if (iter>m_burnIn && iter%m_lag == 0) {
            //accumulate p(w|z)
            for(int i=0; i<this.number_of_topics; i++) {
                for(int v=0; v<this.vocabulary_size; v++) {
                    topic_term_probabilty[i][v] += word_topic_sstat[i][v]; // accumulate the samples during sampling iterations
                }
            }

            for(int k=0; k<number_of_topics; k++)
                m_topics[k] += (m_bitermTopicStat[k]+d_alpha);
            //accumulate p(z|d)
            for(_Doc d:m_trainSet)
                collectStats(d);
        }
    }

    protected void collectStats(_Doc d) {
        if(d instanceof _ParentDoc)
            return;
        for(int k=0; k<this.number_of_topics; k++)
            d.m_topics[k] += d.m_sstat[k];
    }

    @Override
    protected void finalEst() {
        //estimate p(w|z) from all the collected samples
        for(int i=0; i<this.number_of_topics; i++)
            Utils.L1Normalization(topic_term_probabilty[i]);

        Utils.L1Normalization(m_topics);
        //estimate p(z|d) from all the collected samples
        for(_Doc d:m_trainSet)
            estThetaInDoc(d);
    }

    @Override
    protected void estThetaInDoc(_Doc d) {
        if(d instanceof _ParentDoc)
            return;
        Utils.L1Normalization(d.m_topics);
    }

    protected double calculate_log_likelihood(){
        return 0;
    }

}
