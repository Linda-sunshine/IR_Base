package topicmodels.correspondenceModels;

import structures.*;
import utils.Utils;

import java.util.Collection;
import java.util.Arrays;

/**
 * Created by jetcai1900 on 1/13/17.
 */
public class PriorCorrLDA extends corrLDA_Gibbs{

    protected double[] m_alpha_c;
    protected double m_totalAlpha_c;
    protected double d_alpha_c;
    protected double m_mu;

    public PriorCorrLDA(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
                        int number_of_topics, double alpha, double alpha_c, double burnIn, int lag){
        super(number_of_iteration, converge, beta, c, lambda,
         number_of_topics, alpha, burnIn, lag);
        d_alpha_c = alpha_c;
        m_mu = 0.5;
    }

    protected void initialize_probability(Collection<_Doc> collection){
        createSpace();

        m_alpha_c = new double[number_of_topics];
        Arrays.fill(m_alpha_c, d_alpha_c);
        m_totalAlpha_c = Utils.sumOfArray(m_alpha_c);

        for(int i=0; i<number_of_topics; i++){
            Arrays.fill(word_topic_sstat[i], d_beta);
        }
        Arrays.fill(m_sstat, d_beta*vocabulary_size);

        for(_Doc d:collection){
            if(d instanceof _ParentDoc){
                for(_Stn stnObj:d.getSentences()){
                    stnObj.setTopicsVct(number_of_topics);
                }
                d.setTopics4Gibbs(number_of_topics, 0);
            }else if(d instanceof  _ChildDoc){
                ((_ChildDoc)d).setMu(m_mu);
                ((_ChildDoc)d).setTopics4Gibbs_LDA(number_of_topics, 0);
            }

            for(_Word w:d.getWords()){
                word_topic_sstat[w.getTopic()][w.getIndex()] ++;
                m_sstat[w.getTopic()] ++;
            }
        }

        imposePrior();
        m_statisticsNormalized = false;
    }

    public String toString(){
        return String.format("Prior Corr LDA [k:%d, alpha:%.2f, mu: %.2f, beta:%.2f, Gibbs Sampling]",
                number_of_topics, d_alpha, m_mu, d_beta);
    }

    protected double parentChildInfluenceProb(int tid, _ParentDoc d) {
        double term = 1.0;

        if (tid == 0)
            return term;

        for (_ChildDoc cDoc : d.m_childDocs) {
            double muDp = cDoc.getMu() / d.getDocInferLength();
            term *= gammaFuncRatio((int) cDoc.m_sstat[tid], muDp, m_alpha_c[tid]
                    + d.m_sstat[tid] * muDp)
                    / gammaFuncRatio((int) cDoc.m_sstat[0], muDp, m_alpha_c[0]
                    + d.m_sstat[0] * muDp);
        }

        return term;

    }

    protected double gammaFuncRatio(int nc, double muDp, double alphaMuDp) {
        if (nc == 0)
            return 1.0;

        double result = 1.0;
        for (int n = 1; n <= nc; n++) {
            result *= 1 + muDp / (alphaMuDp + n - 1);
        }

        return result;
    }

    protected double childTopicInDocProb(int tid, _ChildDoc d){
        _ParentDoc pDoc = d.m_parentDoc;
        double prob = 0;
        double childTopicSum = Utils.sumOfArray(d.m_sstat);
        double parentTopicSum = Utils.sumOfArray(pDoc.m_sstat);

        double muDp = d.getMu() / parentTopicSum;
        prob = (m_alpha_c[tid] + muDp * pDoc.m_sstat[tid] + d.m_sstat[tid])
                / (m_totalAlpha_c + muDp * parentTopicSum + childTopicSum);

        return prob;
    }

    protected void collectChildStats(_Doc d) {
        _ChildDoc cDoc = (_ChildDoc) d;
        _ParentDoc pDoc = cDoc.m_parentDoc;
        double pDocStatSum = Utils.sumOfArray(pDoc.m_sstat);
        for(int k=0; k<number_of_topics; k++)
            cDoc.m_topics[k] += cDoc.m_sstat[k]+m_alpha_c[k]+cDoc.getMu()*pDoc.m_sstat[k]/pDocStatSum;
    }

    protected double calculate_log_likelihood4Parent(_Doc d) {

        _ParentDoc pDoc = (_ParentDoc) d;
        double docLogLikelihood = 0;

        _SparseFeature[] fv = pDoc.getSparse();

        double docTopicSum = Utils.sumOfArray(pDoc.m_sstat);
        double alphaSum = d_alpha * number_of_topics;

        for(int j=0; j<fv.length; j++){
            int wid = fv[j].getIndex();
            double value = fv[j].getValue();

            double wordLogLikelihood = 0;
            for(int k=0; k<number_of_topics; k++){
                double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)
                        * parentTopicInDocProb(k, pDoc)/ (alphaSum + docTopicSum);
                wordLogLikelihood += wordPerTopicLikelihood;
            }

            if(Math.abs(wordLogLikelihood)<1e-10){
                System.out.println("wordLogLikelihood\t"+wordLogLikelihood);
                wordLogLikelihood += 1e-10;
            }

            wordLogLikelihood = Math.log(wordLogLikelihood);

            docLogLikelihood += value*wordLogLikelihood;
        }

        return docLogLikelihood;
    }

    protected double calculate_log_likelihood4Child(_Doc d) {

        _ChildDoc cDoc = (_ChildDoc) d;
        double docLogLikelihood = 0;

        _SparseFeature[] fv = cDoc.getSparse();

        for(int i=0; i<fv.length; i++){
            int wid = fv[i].getIndex();
            double value = fv[i].getValue();
            double wordLogLikelihood = 0;
            for(int k=0; k<number_of_topics; k++){
                double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
                        * childTopicInDoc(k, cDoc);
                wordLogLikelihood += wordPerTopicLikelihood;
            }

            if(wordLogLikelihood< 1e-10){
                wordLogLikelihood += 1e-10;
                System.out.println("small likelihood in child");
            }

            wordLogLikelihood = Math.log(wordLogLikelihood);

            docLogLikelihood += value*wordLogLikelihood;
        }

        return docLogLikelihood;
    }

    protected double cal_logLikelihood_Perplexity4Parent(_Doc d){
        _ParentDoc pDoc = (_ParentDoc) d;
        double docLogLikelihood = 0.0;

        for (_Word w : pDoc.getWords()) {
            int wid = w.getIndex();

            double wordLogLikelihood = 0;
            for (int k = 0; k < number_of_topics; k++) {
                double wordPerTopicLikelihood = pDoc.m_topics[k]
                        * topic_term_probabilty[k][wid];
                wordLogLikelihood += wordPerTopicLikelihood;
            }
            docLogLikelihood += Math.log(wordLogLikelihood);
        }

        return docLogLikelihood;
    }

    protected double cal_logLikelihood_Perplexity4Child(_Doc d){
        _ChildDoc cDoc = (_ChildDoc)d;
        double docLogLikelihood = 0.0;

        for (_Word w : cDoc.getWords()) {
            int wid = w.getIndex();

            double wordLogLikelihood = 0;
            for (int k = 0; k < number_of_topics; k++) {
                double wordPerTopicLikelihood = cDoc.m_topics[k]
                        * topic_term_probabilty[k][wid];
                wordLogLikelihood += wordPerTopicLikelihood;
            }
            docLogLikelihood += Math.log(wordLogLikelihood);
        }

        return docLogLikelihood;
    }


}
