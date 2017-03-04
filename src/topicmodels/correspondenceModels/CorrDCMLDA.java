package topicmodels.correspondenceModels;

import structures.*;
import utils.Utils;

import java.io.File;
import java.util.Arrays;

/**
 * Created by jetcai1900 on 12/28/16.
 */
public class CorrDCMLDA extends DCMCorrLDA {
    double m_smoothingParam;

    public CorrDCMLDA(int number_of_iteration, double converge, double beta,
                      _Corpus c, double lambda, int number_of_topics, double alpha_a,
                      double alpha_c, double burnIn, double ksi, double tau, int lag,
                      int newtonIter, double newtonConverge){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha_a, alpha_c, burnIn, ksi, tau, lag,
                newtonIter, newtonConverge);

        m_smoothingParam = 0.01;
    }

    public String toString(){
        return String.format("CorrDCMLDA[k:%d, alphaA:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]",number_of_topics, d_alpha, d_beta,
                1 - m_testWord4PerplexityProportion);
    }

    protected double parentChildInfluenceProb(int tid, _ParentDoc4DCM d){
        double term = 1.0;

        if(tid == 0)
            return term;

        for(_ChildDoc cDoc:d.m_childDocs){
            term *= influenceRatio(cDoc.m_sstat[tid], d.m_sstat[tid], cDoc.m_sstat[0], d.m_sstat[0]);
        }

        return term;
    }

    protected double influenceRatio(double njc, double njp, double n1c, double n1p){
        double ratio = 1.0;

        for(int n=1; n<=n1c; n++){
            ratio *= (n1p+m_smoothingParam)*1.0/(n1p+1+m_smoothingParam);
        }

        for(int n=1; n<=njc; n++){
            ratio *= (njp+1+m_smoothingParam)*1.0/(njp+m_smoothingParam);
        }

        return ratio;
    }

    protected double childTopicInDocProb(int tid, _ChildDoc d, _ParentDoc4DCM pDoc){
        double prob = 0;
        double parentTopicSum = Utils.sumOfArray(pDoc.m_sstat);

        prob = (pDoc.m_sstat[tid]+m_smoothingParam)/(parentTopicSum+m_smoothingParam*number_of_topics);
        return prob;
    }

    public void updateParameter(int iter, File weightFolder){
        File weightIterFolder = new File(weightFolder, "_" + iter);
        if (!weightIterFolder.exists()) {
            weightIterFolder.mkdir();
        }

        initialAlphaBeta();
        updateAlpha();

        for (int k = 0; k < number_of_topics; k++)
            updateBeta(k);

        for (int k = 0; k < number_of_topics; k++)
            m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);

        String fileName = iter + ".txt";
//		saveParameter2File(weightIterFolder, fileName);

    }

    protected void collectStats(_Doc d){
        if(d instanceof _ParentDoc4DCM){
            _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
            for(int k=0; k<number_of_topics; k++){
                pDoc.m_topics[k] += pDoc.m_sstat[k]+m_alpha[k];
                for(int v=0; v<vocabulary_size; v++){
                    pDoc.m_wordTopic_prob[k][v] += pDoc.m_wordTopic_stat[k][v]+m_beta[k][v];
                }
            }
        }else if(d instanceof _ChildDoc){
            _ChildDoc cDoc = (_ChildDoc)d;
            _ParentDoc pDoc = cDoc.m_parentDoc;
            double topicSum = Utils.sumOfArray(pDoc.m_sstat);
            double muDp = cDoc.getMu()/topicSum;
            for(int k=0; k<number_of_topics; k++){
                cDoc.m_topics[k] += cDoc.m_sstat[k];
            }
        }
    }

    protected double calculate_log_likelihood(_ParentDoc4DCM d){
        double docLogLikelihood = 0;
        int docID = d.getID();

        double parentDocLength = d.getTotalDocLength();

        for(int k=0; k<number_of_topics; k++){
            double term = Utils.lgamma(d.m_sstat[k]+m_alpha[k]);
            docLogLikelihood += term;

            term = Utils.lgamma(m_alpha[k]);
            docLogLikelihood -= term;
        }

        docLogLikelihood += Utils.lgamma(m_totalAlpha);
        docLogLikelihood -= Utils.lgamma(parentDocLength+m_totalAlpha);

        for(int k=0; k<number_of_topics; k++){
            for(int v=0; v<vocabulary_size; v++){
                double term = Utils.lgamma(d.m_wordTopic_stat[k][v]+m_beta[k][v]);
                docLogLikelihood += term;

                term = Utils.lgamma(m_beta[k][v]);
                docLogLikelihood -= term;
            }

            docLogLikelihood += Utils.lgamma(m_totalBeta[k]);
            docLogLikelihood -= Utils.lgamma(d.m_topic_stat[k]+m_totalBeta[k]);
        }

        for(_ChildDoc cDoc:d.m_childDocs){
           double term = 0;
            for(int k=0; k<number_of_topics; k++){
                double term1 = 0;
                term1 += cDoc.m_sstat[k]*(Math.log(d.m_sstat[k]+m_smoothingParam)-Math.log(parentDocLength+m_smoothingParam*number_of_topics));
                term += term1;
            }
            docLogLikelihood += term;
        }

        return docLogLikelihood;

    }

    protected void estThetaInDoc(_Doc d){
        if (d instanceof _ParentDoc4DCM) {
            for (int i = 0; i < number_of_topics; i++)
                Utils.L1Normalization(((_ParentDoc4DCM) d).m_wordTopic_prob[i]);
        }
        Utils.L1Normalization(d.m_topics);

    }

    protected double cal_logLikelihood_partial4Parent(_Doc d) {
        _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
        double docLogLikelihood = 0.0;

        for (_Word w : pDoc.getTestWords()) {
            int wid = w.getIndex();

            double wordLogLikelihood = 0;
            for (int k = 0; k < number_of_topics; k++) {
                double wordPerTopicLikelihood = pDoc.m_topics[k]
                        *pDoc.m_wordTopic_prob[k][wid];
                wordLogLikelihood += wordPerTopicLikelihood;
            }
            docLogLikelihood += Math.log(wordLogLikelihood);
        }

        return docLogLikelihood;
    }

    protected double cal_logLikelihood_partial4Child(_Doc d) {
        double docLogLikelihood = 0.0;

        _ChildDoc cDoc = (_ChildDoc)d;
        _ParentDoc4DCM pDoc = (_ParentDoc4DCM) cDoc.m_parentDoc;

        for (_Word w : cDoc.getTestWords()) {
            int wid = w.getIndex();

            double wordLogLikelihood = 0;
            for (int k = 0; k < number_of_topics; k++) {
                double wordPerTopicLikelihood = cDoc.m_topics[k]
                        * pDoc.m_wordTopic_prob[k][wid];
                wordLogLikelihood += wordPerTopicLikelihood;
            }
            docLogLikelihood += Math.log(wordLogLikelihood);
        }

        return docLogLikelihood;
    }



}
