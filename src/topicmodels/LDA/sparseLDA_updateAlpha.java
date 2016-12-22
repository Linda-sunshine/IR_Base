package topicmodels.LDA;

import structures._Corpus;
import structures._Doc;
import structures._Doc4SparseDCMLDA;
import structures._Word;
import utils.Utils;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;

/**
 * Created by jetcai1900 on 12/13/16.
 */
public class sparseLDA_updateAlpha extends sparseLDA{
    double[] m_alpha;
    double m_totalAlpha;
    int m_newtonIter;
    double m_newtonConverge;

    public sparseLDA_updateAlpha(int number_of_iteration, double converge, double beta,
                                 _Corpus c, double lambda,
                                 int number_of_topics, double alpha, double burnIn, int lag, double tParam, double sParam, int newtonIter, double newtonConverge){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics,  alpha, burnIn, lag, tParam, sParam);

        m_newtonIter = newtonIter;
        m_newtonConverge = newtonConverge;
    }

    public String toString() {
        return String.format("sparseLDA updateAlpha[k:%d, alpha:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta, 1-m_testWord4PerplexityProportion);
    }

    protected void initialAlpha(){
        m_alpha = new double[number_of_topics];
        m_totalAlpha = 0;

        for(int k=0; k<number_of_topics; k++){
            m_alpha[k] = 1.0/number_of_topics;
            m_totalAlpha += m_alpha[k];
        }
    }

    protected void initialize_probability(Collection<_Doc> collection) {
        initialAlpha();

        for(int i=0; i<number_of_topics; i++)
            Arrays.fill(word_topic_sstat[i], d_beta);
        Arrays.fill(m_sstat, d_beta*vocabulary_size);
        Arrays.fill(m_topicProbCache, 0);

        // initialize topic-word allocation, p(w|z)
        for(_Doc d:collection) {
            _Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;

            DCMDoc.setTopics4Gibbs(number_of_topics, m_alpha);//allocate memory and randomize it
            for(_Word w:d.getWords()) {
                word_topic_sstat[w.getTopic()][w.getIndex()] ++;
                m_sstat[w.getTopic()] ++;
            }
        }

        imposePrior();
    }

    public void EM() {
        System.out.format("Starting %s...\n", toString());

        long starttime = System.currentTimeMillis();

        m_collectCorpusStats = true;
        initialize_probability(m_trainSet);

        String filePrefix = "./data/results/sparseLDA";
        File weightFolder = new File(filePrefix + "");
        if (!weightFolder.exists()) {
            // System.out.println("creating directory for weight"+weightFolder);
            weightFolder.mkdir();
        }

        double delta = 0, last = 0, current = 0;
        int i = 0, displayCount = 0;
        do {

            long startTime = System.currentTimeMillis();
            for (int j = 0; j < number_of_iteration; j++) {
                init();
//				System.out.println("iteration\t" + j);
                for (_Doc d : m_trainSet)
                    calculate_E_step(d);
            }
            long endTime = System.currentTimeMillis();

            System.out.println("per iteration e step time\t"
                    + (endTime - startTime) / 1000);

            startTime = System.currentTimeMillis();
            updateParameter(i, weightFolder);
            endTime = System.currentTimeMillis();

            System.out.println("per iteration m step time\t"
                    + (endTime - startTime) / 1000);

            if (m_converge > 0
                    || (m_displayLap > 0 && i % m_displayLap == 0 && displayCount > 6)) {
                // required to display log-likelihood
                current = calculate_log_likelihood();
                // together with corpus-level log-likelihood

                if (i > 0)
                    delta = (last - current) / last;
                else
                    delta = 1.0;
                last = current;
            }

            if (m_displayLap > 0 && i % m_displayLap == 0) {
                if (m_converge > 0) {
                    System.out.format(
                            "Likelihood %.3f at step %s converge to %f...\n",
                            current, i, delta);
                    infoWriter.format(
                            "Likelihood %.3f at step %s converge to %f...\n",
                            current, i, delta);

                } else {
                    System.out.print(".");
                    if (displayCount > 6) {
                        System.out.format("\t%d:%.3f\n", i, current);
                        infoWriter.format("\t%d:%.3f\n", i, current);
                    }
                    displayCount++;
                }
            }

            if (m_converge > 0 && Math.abs(delta) < m_converge)
                break;// to speed-up, we don't need to compute likelihood in
            // many cases
        } while (++i < this.number_of_iteration);


        finalEst();

        long endtime = System.currentTimeMillis() - starttime;
        System.out
                .format("Likelihood %.3f after step %s converge to %f after %d seconds...\n",
                        current, i, delta, endtime / 1000);
        infoWriter
                .format("Likelihood %.3f after step %s converge to %f after %d seconds...\n",
                        current, i, delta, endtime / 1000);
    }

    protected double calculate_log_likelihood() {
        double logLikelihood = 0.0;
        for (_Doc d : m_trainSet) {
            logLikelihood += calculate_log_likelihood(d);
        }
        return logLikelihood;
    }

    protected double topicInDocProb(int tid, double denominator, _Doc4SparseDCMLDA d){
        double term1 = d.m_sstat[tid];
        term1 += m_alpha[tid];

        return term1/denominator;
    }

    protected void sampleOnOffIndicator(_Doc4SparseDCMLDA DCMDoc){
        for(int k=0; k<number_of_topics; k++){

            boolean xk = DCMDoc.m_topicIndicator[k];
            if(xk==true){
                DCMDoc.m_indicatorTrue_stat --;
                DCMDoc.m_alphaDoc -= m_alpha[k];
            }

            if(DCMDoc.m_sstat[k]>0){
                xk = true;
            }else{
                double prob = 0;

                double trueProb = 0;
                double falseProb = 0;
                double term1 = DCMDoc.m_alphaDoc;
                double term2 = m_alpha[k];
                double term3 = m_s + DCMDoc.m_indicatorTrue_stat;
                double term4 = m_t + number_of_topics-1
                        - DCMDoc.m_indicatorTrue_stat;
                //double term1 = DCMDoc.m_alphaDoc+m_alpha[k], DCMDoc.m_alphaDoc+m_alpha[k]+DCMDoc.getTotalDocLength());
                //double term2 = (m_s+DCMDoc.m_indicatorTrue_stat);
                double Q = term3 / term4;
                for (int i = 0; i < DCMDoc.getTotalDocLength(); i++) {
                    double QTemp = (term1 + i) / (term1 + term2 + i);
                    Q *= QTemp;
                }

                falseProb = 1.0/(Q+1);
                trueProb = 1-falseProb;

//                System.out.println("falseProb:\t"+falseProb);

                prob = m_rand.nextDouble()*(trueProb+falseProb);
                if(prob<trueProb)
                    xk = true;
                else
                    xk = false;
            }

            DCMDoc.m_topicIndicator[k] = xk;
            if(xk==true){
                DCMDoc.m_indicatorTrue_stat++;
                DCMDoc.m_alphaDoc += m_alpha[k];
            }

        }
    }

    protected void updateParameter(int iter, File weightFolder) {

        File weightIterFolder = new File(weightFolder, "_" + iter);
        if (!weightIterFolder.exists()) {
            weightIterFolder.mkdir();
        }

        initialAlpha();
        updateAlpha();

        String fileName = iter + ".txt";
        saveParameter2File(weightIterFolder, fileName);

    }

    protected void updateAlpha(){
        double diff = 0;
        double smallAlpha = 0.1;

        int iteration = 0;
        do {

            diff = 0;
            double[] wordNum4Tid = new double[number_of_topics];

            double totalAlphaDenominator = 0;

            double deltaAlpha = 0;

            for (int k = 0; k < number_of_topics; k++) {
                wordNum4Tid[k] = 0;
                double totalAlphaNumerator = 0;
                totalAlphaDenominator = 0;
                for (_Doc d : m_trainSet) {
                    _Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;
                    if(DCMDoc.m_topicIndicator[k]==false)
                        continue;

                    wordNum4Tid[k] += DCMDoc.m_sstat[k];

                    totalAlphaDenominator += Utils.digamma(DCMDoc.getTotalDocLength()+DCMDoc.m_alphaDoc)-Utils.digamma(DCMDoc.m_alphaDoc);
                    totalAlphaNumerator += Utils.digamma(m_alpha[k]
                            + d.m_sstat[k])
                            - Utils.digamma(m_alpha[k]);
                }

                if(wordNum4Tid[k]==0){
                    deltaAlpha = 0;
                }else{
                    deltaAlpha = totalAlphaNumerator*1.0/totalAlphaDenominator;
                }

                double newAlpha = m_alpha[k] * deltaAlpha+d_alpha;
                double t_diff = Math.abs(m_alpha[k] - newAlpha);
                if (t_diff > diff)
                    diff = t_diff;

                m_alpha[k] = newAlpha;
            }

            iteration++;

            if(iteration > m_newtonIter)
                break;

        }while(diff>m_newtonConverge);

        m_totalAlpha = 0;
        for (int k = 0; k < number_of_topics; k++) {
            m_totalAlpha += m_alpha[k];
        }

        for(_Doc d:m_trainSet){
            _Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;
            DCMDoc.m_alphaDoc = 0;
            for(int k=0; k<number_of_topics; k++){
                if(DCMDoc.m_topicIndicator[k]==true)
                    DCMDoc.m_alphaDoc += m_alpha[k];
            }

        }
    }

    protected void saveParameter2File(File fileFolder, String fileName) {
        try {
            File paramFile = new File(fileFolder, fileName);

            PrintWriter pw = new PrintWriter(paramFile);
            pw.println("alpha");
            for (int k = 0; k < number_of_topics; k++) {
                pw.print(m_alpha[k] + "\t");
            }

            pw.flush();
            pw.close();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    protected void finalEst(){
        runLastEM();
        for (_Doc d : m_trainSet) {
            estThetaInDoc(d);
        }
        estGlobalParameter();
    }

    protected void runLastEM(){

        for (int j = 0; j < number_of_iteration; j++) {
            init();
            for (_Doc d : m_trainSet) {
                _Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;

                calculate_E_step(DCMDoc);
                if (j % 20 == 0) {
                    DCMDoc.m_MStepIter += 1;

                    for (int k = 0; k < number_of_topics; k++)
                        if (DCMDoc.m_topicIndicator[k] == true) {
                            DCMDoc.m_topicIndicator_prob[k] += 1; // miss m_s
                        }

                    DCMDoc.m_topicIndicator_distribution += DCMDoc.m_indicatorTrue_stat;
                }
            }

        }

        collectStats();
    }

    protected void collectStats() {
        for(int k=0; k<number_of_topics; k++)
            for(int v=0; v<vocabulary_size; v++)
                topic_term_probabilty[k][v] = word_topic_sstat[k][v];

        for(_Doc d:m_trainSet){
            _Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;

            for (int k = 0; k < this.number_of_topics; k++) {
                if(DCMDoc.m_topicIndicator[k]==false)
                    continue;
                DCMDoc.m_topics[k] = DCMDoc.m_sstat[k] + m_alpha[k];
            }
        }
    }


    protected void estGlobalParameter(){
        for(int i=0; i<number_of_topics; i++)
            Utils.L1Normalization(topic_term_probabilty[i]);
    }

    protected double calculate_log_likelihood(_Doc d) {
        double docLogLikelihood = 0.0;

        _Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;

        for (int k = 0; k < number_of_topics; k++) {
            if(DCMDoc.m_topicIndicator[k]==false)
                continue;
            double term = Utils.lgamma(DCMDoc.m_sstat[k] + m_alpha[k]);
            docLogLikelihood += term;

            term = Utils.lgamma(m_alpha[k]);
            docLogLikelihood -= term;

        }

        docLogLikelihood += Utils.lgamma(DCMDoc.m_alphaDoc);
        docLogLikelihood -= Utils.lgamma(DCMDoc.getTotalDocLength() + DCMDoc.m_alphaDoc);


        double totalBeta = d_beta*vocabulary_size;
        for (int k = 0; k < number_of_topics; k++) {
            for (int v = 0; v < vocabulary_size; v++) {
                double term = Utils.lgamma(word_topic_sstat[k][v]);
                docLogLikelihood += term;

                term = Utils.lgamma(d_beta);
                docLogLikelihood -= term;

            }

            docLogLikelihood += Utils.lgamma(totalBeta);
            docLogLikelihood -= Utils.lgamma(m_sstat[k]+totalBeta);
        }

        docLogLikelihood += Utils.lgamma(m_t+m_s)-Utils.lgamma(m_t)-Utils.lgamma(m_s);
        docLogLikelihood += Utils.lgamma(DCMDoc.m_indicatorTrue_stat+m_s)+Utils.lgamma(m_t+number_of_topics-DCMDoc.m_indicatorTrue_stat)-Utils.lgamma(m_t+m_s+number_of_topics);

        return docLogLikelihood;
    }
}
