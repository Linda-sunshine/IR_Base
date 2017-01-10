package topicmodels.correspondenceModels;

import LBFGS.LBFGS;
import topicmodels.LDA.LDA_Variational;
import structures._Corpus;
import structures._Doc;
import structures._ChildDoc;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import structures._ParentDoc4DCM;
import structures._Stn;
import structures._SparseFeature;
import utils.Utils;
import java.util.Arrays;
import java.util.HashMap;
import structures._ParentDoc;


/**
 * Created by jetcai1900 on 12/17/16.
 */
public class weightedCorrespondenceModel extends LDA_Variational {

    protected double[][] m_beta;
    protected double[] m_alpha_c;
    protected double[] m_alpha_stat;
    protected double[] m_alpha_c_stat;
    protected double[][] m_beta_stat;
    protected int m_parentDocNum;
    protected int m_childDocNum;
    protected double m_lbfgsConverge;

    public weightedCorrespondenceModel(int number_of_iteration, double converge, double beta,
                                       _Corpus c, double lambda, int number_of_topics, double alpha,
                                       int varMaxIter, double varConverge, double lbfgsConverge){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha,
                varMaxIter, varConverge);
        m_varConverge = varConverge;
        m_varMaxIter = varMaxIter;
        m_logSpace = true;
        m_lbfgsConverge = lbfgsConverge;

    }

    public void createSpace(){
        super.createSpace();

        m_parentDocNum = 0;
        m_childDocNum = 0;

        m_alpha_c  = new double[number_of_topics];

        for(int k=0; k<number_of_topics; k++) {
            m_alpha_c[k] = d_alpha;
        }

    }

    public String toString(){
        return String.format("WCM, Variational Inference[k:%d, alpha:%.2f, beta:%.2f]",number_of_topics, d_alpha, d_beta);
    }

    protected void initialize_probability(Collection<_Doc> collection){
        init();

        for(int k=0; k<number_of_topics; k++){
            Arrays.fill(topic_term_probabilty[k], d_beta);
        }

        for(_Doc d:collection){
            if(d instanceof _ParentDoc4DCM){
                int totalWords = 0;


                m_parentDocNum += 1;
                _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
                pDoc.setTopics4Variational(number_of_topics, d_alpha, vocabulary_size, d_beta);

                totalWords += pDoc.getTotalDocLength();
                for(_Stn stnObj:pDoc.getSentences()){
                    stnObj.setTopicsVct(number_of_topics);
                }

                collectStats4Parent((_ParentDoc4DCM) d);

                for(_ChildDoc cDoc:pDoc.m_childDocs){
                    totalWords += cDoc.getTotalDocLength();
                    m_childDocNum += 1;
                    cDoc.setTopics4Variational(number_of_topics, d_alpha);
//                    collectStats4Child(cDoc);
                }
            }
        }
//        calculate_M_step(0);
//        imposePrior();
    }

    protected void init(){
        m_alpha_stat = new double[number_of_topics];
        m_alpha_c_stat = new double[number_of_topics];
        word_topic_sstat = new double[number_of_topics][vocabulary_size];

        for(int k=0; k<number_of_topics; k++){
            m_alpha_stat[k] = 0;

            m_alpha_c_stat[k] =0;

            Arrays.fill(word_topic_sstat[k], d_beta-1);
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

            for(_Doc d:m_trainSet)
                calculate_E_step(d);

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

            if (m_converge>0) {
                System.out.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
                infoWriter.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);

            }


            if (m_converge>0 && Math.abs(delta)<m_converge)
                break;//to speed-up, we don't need to compute likelihood in many cases
        } while (++i<this.number_of_iteration);

        finalEst();

        long endtime = System.currentTimeMillis() - starttime;
        System.out.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);
        infoWriter.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);

    }

    protected void imposePrior() {
        if (word_topic_prior != null) {
            for (int k = 0; k < number_of_topics; k++) {
                for (int v = 0; v < vocabulary_size; v++) {
                    m_beta[k][v] = word_topic_prior[k][v];
                }
            }
        }
    }

    public void LoadPrior(String fileName, double eta) {
        if (fileName == null || fileName.isEmpty()) {
            return;
        }

        try {

            if (word_topic_prior == null) {
                word_topic_prior = new double[number_of_topics][vocabulary_size];
            }

            for (int k = 0; k < number_of_topics; k++)
                Arrays.fill(word_topic_prior[k], 0);

            String tmpTxt;
            String[] lineContainer;
            String[] featureContainer;
            int tid = 0;

            HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
            for (int i = 0; i < m_corpus.getFeatureSize(); i++) {
                featureNameIndex.put(m_corpus.getFeature(i),
                        featureNameIndex.size());
            }

            BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(fileName), "UTF-8"));

            while ((tmpTxt = br.readLine()) != null) {
                tmpTxt = tmpTxt.trim();
                if (tmpTxt.isEmpty())
                    continue;

                lineContainer = tmpTxt.split("\t");

                tid = Integer.parseInt(lineContainer[0]);
                for (int i = 1; i < lineContainer.length; i++) {
                    featureContainer = lineContainer[i].split(":");

                    String featureName = featureContainer[0];
                    double featureProb = Double
                            .parseDouble(featureContainer[1]);

                    int featureIndex = featureNameIndex.get(featureName);

                    word_topic_prior[tid][featureIndex] = featureProb;
                }
            }

            System.out.println("prior is added");
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public double calculate_E_step(_Doc d){
        if(d instanceof _ChildDoc)
            return;

        _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;

        double last = 1;
        if(m_varConverge>0)
            last = calculate_log_likelihood(pDoc);

        double current = last, converge, logSum, wVal;
        int iter=0, wID;

        do{
            updateEta4Parent(pDoc);
            updateGamma4Parent(pDoc);
//            updateLambda(pDoc);
            updateEta4Child(pDoc);
            updatePi4Child(pDoc);
//            updateZeta4Child(pDoc);
//            updateLambda(pDoc);

            if(m_varConverge>0){
                current = calculate_log_likelihood(pDoc);
                converge = Math.abs((current-last)/last);
                last = current;

                if(converge<m_varConverge)
                    break;
            }
        }while(++iter<m_varMaxIter);

        collectStats(pDoc);

        return current;

    }



    protected void collectStats4Parent(_ParentDoc4DCM pDoc){

        _SparseFeature[] pDocFV = pDoc.getSparse();
        for(int n=0; n<pDocFV.length; n++){
            _SparseFeature fv = pDocFV[n];
            int wID = fv.getIndex();
            double wVal = fv.getValue();
            for(int k=0; k<number_of_topics; k++){
                word_topic_sstat[k][wID] += wVal*pDoc.m_phi[n][k];
            }
        }

        double gammaSum = Utils.sumOfArray(pDoc.m_sstat);
        for(int k=0; k<number_of_topics; k++) {
            m_alpha_stat[k] += Utils.digamma(pDoc.m_sstat[k]) - Utils.digamma(gammaSum);
        }


    }

    protected void collectStats4Child(_ChildDoc cDoc){

//            double piSum = Utils.sumOfArray(cDoc.m_sstat);
//            for (int k = 0; k < number_of_topics; k++)
//                m_alpha_c_stat[k] += Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum);
        double piSum = Utils.sumOfArray(cDoc.m_sstat);
        for (int k = 0; k < number_of_topics; k++)
            m_alpha_stat[k] += Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum);

        _SparseFeature[] cDocFV = cDoc.getSparse();
        for(int n=0; n< cDocFV.length; n++){
            int wID = cDocFV[n].getIndex();
            double wVal = cDocFV[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                word_topic_sstat[k][wID] += wVal*cDoc.m_phi[n][k];
            }
        }

    }

    protected void collectStats(_ParentDoc4DCM pDoc){

        _SparseFeature[] pDocFV = pDoc.getSparse();
        for(int n=0; n<pDocFV.length; n++){
            _SparseFeature fv = pDocFV[n];
            int wID = fv.getIndex();
            double wVal = fv.getValue();
            for(int k=0; k<number_of_topics; k++){
                word_topic_sstat[k][wID] += wVal*pDoc.m_phi[n][k];
            }
        }

        double gammaSum = Utils.sumOfArray(pDoc.m_sstat);
        for(int k=0; k<number_of_topics; k++) {
            m_alpha_stat[k] += Utils.digamma(pDoc.m_sstat[k]) - Utils.digamma(gammaSum);
        }

        for(_ChildDoc cDoc:pDoc.m_childDocs) {

//            double piSum = Utils.sumOfArray(cDoc.m_sstat);
//            for (int k = 0; k < number_of_topics; k++)
//                m_alpha_c_stat[k] += Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum);
            double piSum = Utils.sumOfArray(cDoc.m_sstat);
            for (int k = 0; k < number_of_topics; k++)
                m_alpha_stat[k] += Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum);
                // m_alpha_c_stat[k] += Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum);

                // m_alpha[k] += Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum);


            _SparseFeature[] cDocFV = cDoc.getSparse();
            for(int n=0; n< cDocFV.length; n++){
                int wID = cDocFV[n].getIndex();
                double wVal = cDocFV[n].getValue();
                for(int k=0; k<number_of_topics; k++){
                    word_topic_sstat[k][wID] += wVal*cDoc.m_phi[n][k];
                }
            }
        }
    }

    public void updateEta4Parent(_ParentDoc4DCM pDoc){
        _SparseFeature[] fvs = pDoc.getSparse();
        for(int n=0; n<fvs.length; n++){
            int wID = fvs[n].getIndex();
            double wVal = fvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                pDoc.m_phi[n][k] = Utils.digamma(pDoc.m_sstat[k])+Math.log(topic_term_probabilty[k][wID]);
            }

            double logSum = logSumOfExponentials(pDoc.m_phi[n]);
            double phiSum = 0;

            for(int k=0; k<number_of_topics; k++){
                pDoc.m_phi[n][k] = Math.exp(pDoc.m_phi[n][k] - logSum);
//
//                if((pDoc.m_phi[n][k]-logSum)<-200){
//                    pDoc.m_phi[n][k] = 1e-20;
//                    System.out.println("too small");
//                }else {
//                    pDoc.m_phi[n][k] = Math.exp(pDoc.m_phi[n][k] - logSum);
//                }
                phiSum += pDoc.m_phi[n][k];
            }

            if(Math.abs(phiSum-1)>1)
                System.out.println("phiSum for article\t"+phiSum);
        }
    }

    public void updateGamma4Parent(_ParentDoc4DCM pDoc){
        _SparseFeature[] pDocFV = pDoc.getSparse();
        for(int k=0; k<number_of_topics; k++)
            pDoc.m_sstat[k] = m_alpha[k];

        for(int n=0; n<pDocFV.length; n++){
            _SparseFeature fv = pDocFV[n];
            int wID = fv.getIndex();
            double wVal = fv.getValue();

            for(int k=0; k<number_of_topics; k++){
                pDoc.m_sstat[k] += wVal*pDoc.m_phi[n][k];
            }
        }
    }

//    public void updateGamma4Parent(_ParentDoc4DCM pDoc){
//        int[] iflag = {0}, iprint={-1,3};
//        double fValue = 0;
//        int fSize = pDoc.m_sstat.length;
//
//        double[] gammaGradient = new double[fSize];
//        Arrays.fill(gammaGradient,0);
//
//        double[] gammaDiag = new double[fSize];
//        Arrays.fill(gammaDiag, 0);
//
//        double[] gamma = new double[fSize];
//        double[] oldGamma = new double[fSize];
//
//        for(int k=0; k<fSize; k++){
//            gamma[k] = Math.log(pDoc.m_sstat[k]);
//            oldGamma[k] = Math.log(pDoc.m_sstat[k]);
//        }
//
////        double diff = 0;
//
//        try{
//            do{
//                double diff = 0;
//                fValue = gammaFuncGradientVal(pDoc, gamma, gammaGradient);
//                LBFGS.lbfgs(fSize, 4, gamma, fValue, gammaGradient, false, gammaDiag, iprint, 1e-2, 1e-10, iflag);
//
//                for(int k=0; k<fSize; k++){
//                    double tempDiff = 0;
//                    tempDiff = gamma[k]-oldGamma[k];
//                    if(Math.abs(tempDiff)>diff){
//                        diff = Math.abs(tempDiff);
//                    }
//                    oldGamma[k] = gamma[k];
//                }
//
//                if(diff<m_lbfgsConverge){
////                    System.out.print("diff\t"+diff+"finish update gamma");
//                    break;
//                }
//
//            }while(iflag[0]!=0);
//        }catch(LBFGS.ExceptionWithIflag e){
//            e.printStackTrace();
//        }
//
//        for(int k=0; k<fSize; k++){
//            pDoc.m_sstat[k] = Math.exp(gamma[k]);
//
////            System.out.println(pDoc.getName()+"\tpDoc.m_sstat[]"+pDoc.m_sstat[k]);
//        }
//
//    }

    public double gammaFuncGradientVal(_ParentDoc4DCM pDoc, double[]gamma, double[] gammaGradient){

        Arrays.fill(gammaGradient, 0);
        double funcVal = 0;

        double[] expGamma = new double[number_of_topics];
        double expGammaSum = 0;

        for(int k=0; k<number_of_topics; k++){
            expGamma[k] = Math.exp(gamma[k]);
            expGammaSum += expGamma[k];
        }

        funcVal -= Utils.lgamma(expGammaSum);

        double constantGradient = 0;
        for(int k=0; k<number_of_topics; k++){
            constantGradient += (m_alpha[k]-expGamma[k])*Utils.trigamma(expGammaSum);
        }

        for(int k=0; k<number_of_topics; k++){
            gammaGradient[k] = (m_alpha[k]-expGamma[k])*Utils.trigamma(expGamma[k]);
            funcVal += (m_alpha[k]-1)*(Utils.digamma(expGamma[k])-Utils.digamma(expGammaSum));
            funcVal += Utils.lgamma(expGamma[k]);
            funcVal -= (expGamma[k]-1)*(Utils.digamma(expGamma[k])-Utils.digamma(expGammaSum));
        }

        _SparseFeature[] fvs = pDoc.getSparse();
        for(int n=0; n<fvs.length; n++){
            int wID = fvs[n].getIndex();
            double wVal = fvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                funcVal += pDoc.m_phi[n][k]*wVal*(Utils.digamma(expGamma[k])-Utils.digamma(expGammaSum));

                gammaGradient[k] += pDoc.m_phi[n][k]*wVal*Utils.trigamma(expGamma[k]);
                constantGradient += pDoc.m_phi[n][k]*wVal*Utils.trigamma(expGammaSum);
//                gammaGradient[k] -= pDoc.m_phi[n][k]*wVal*Utils.trigamma(expGammaSum);
            }
        }


        for(_ChildDoc cDoc:pDoc.m_childDocs){

            double piSum = Utils.sumOfArray(cDoc.m_sstat);

            _SparseFeature[] cDocFvs = cDoc.getSparse();
            for(int n=0; n<cDocFvs.length; n++) {
                int wID = cDocFvs[n].getIndex();
                double wVal = cDocFvs[n].getValue();
                for(int k=0; k<number_of_topics; k++){
                    funcVal += cDoc.m_phi[n][k]*wVal*(Utils.digamma(expGamma[k])-Utils.digamma(expGammaSum));
                    funcVal -= cDoc.m_phi[n][k]*wVal*(Utils.dotProduct(cDoc.m_sstat, expGamma))/(piSum*expGammaSum*cDoc.m_zeta);

                    gammaGradient[k] += cDoc.m_phi[n][k]*wVal*(Utils.trigamma(expGamma[k]));
                    constantGradient += cDoc.m_phi[n][k]*wVal*Utils.trigamma(expGammaSum);
//                    gammaGradient[k] -= cDoc.m_phi[n][k]*wVal*Utils.trigamma(expGammaSum);
                    double temp = cDoc.m_sstat[k]*expGammaSum-Utils.dotProduct(cDoc.m_sstat, expGamma);
                    gammaGradient[k] -= cDoc.m_phi[n][k]*wVal*temp/(piSum*expGammaSum*expGammaSum*cDoc.m_zeta);
                }

            }
        }

        for(int k=0; k<number_of_topics; k++){
            gammaGradient[k] -= constantGradient;
            gammaGradient[k] *= expGamma[k];
            gammaGradient[k] = 0-gammaGradient[k];
        }

        return -funcVal;
    }

    public void updateLambda(_ParentDoc4DCM pDoc){
       _SparseFeature[] fvs = pDoc.getSparse();

        int totalWord = 0;
        double totalLambda = 0;

        for(int k=0; k<number_of_topics; k++)
            for(int v=0; v<vocabulary_size; v++)
                pDoc.m_lambda_stat[k][v] = m_beta[k][v];

        for(int n=0; n<fvs.length; n++){
            int wID = fvs[n].getIndex();
            double wVal = fvs[n].getValue();
            totalWord += wVal;

            double phiSum = Utils.sumOfArray(pDoc.m_phi[n]);
            if(Math.abs(phiSum-1)>1){
                System.out.println("inequal to 1\t"+n+phiSum);
            }

            for(int k=0; k<number_of_topics; k++){
                pDoc.m_lambda_stat[k][wID] += wVal*pDoc.m_phi[n][k];

                if(Double.isNaN(pDoc.m_lambda_stat[k][wID])){
                    System.out.println("nan error article\t"+n+" "+wID);
                }
            }
        }

        for(_ChildDoc cDoc:pDoc.m_childDocs){
            _SparseFeature[] cFvs = cDoc.getSparse();
            for(int n=0; n<cFvs.length; n++){
                int wID = cFvs[n].getIndex();
                double wVal = cFvs[n].getValue();

                totalWord += wVal;

                double phiSum = Utils.sumOfArray(cDoc.m_phi[n]);
                if(Math.abs(phiSum-1)>1){
                    System.out.println("inequal to 1\t"+n+phiSum);
                    for(int k=0; k<number_of_topics; k++){
                        System.out.println("\t\t"+cDoc.m_phi[n][k]);
                    }
                }

                for(int k=0; k<number_of_topics; k++){
                    pDoc.m_lambda_stat[k][wID] += wVal*cDoc.m_phi[n][k];

                    if(Double.isNaN(pDoc.m_lambda_stat[k][wID])){
                        System.out.println("nan error comment\t"+n+" "+wID);
                    }
                }
            }
        }

        for(int k=0; k<number_of_topics; k++) {
            pDoc.m_lambda_topicStat[k] = Utils.sumOfArray(pDoc.m_lambda_stat[k]);
            totalLambda += pDoc.m_lambda_topicStat[k];
        }

//        System.out.println("total Words in this doc\t"+pDoc.getName()+"\t"+totalWord+"\t"+totalLambda);
    }

    public void updatePi4Child(_ChildDoc cDoc){
//        for(_ChildDoc cDoc:pDoc.m_childDocs){
            for(int k=0; k<number_of_topics; k++){
                cDoc.m_sstat[k] = m_alpha[k];
            }
            _SparseFeature[] cDocFv = cDoc.getSparse();
            for(int n=0; n<cDocFv.length; n++){
                _SparseFeature fv = cDocFv[n];
                int wID = fv.getIndex();
                double wVal = fv.getValue();

                for(int k=0; k<number_of_topics; k++){
                    cDoc.m_sstat[k] += wVal*cDoc.m_phi[n][k];
                }
            }
//        }

    }

//    public void updatePi4Child(_ParentDoc4DCM pDoc){
//        double gammaSum = Utils.sumOfArray(pDoc.m_sstat);
//        for(_ChildDoc cDoc:pDoc.m_childDocs) {
//            int[] iflag = {0}, iprint = {-1, 3};
//            double fValue = 0;
//            int fSize = cDoc.m_sstat.length;
//
//            double[] piGradient = new double[fSize];
//            Arrays.fill(piGradient, 0);
//
//            double[] piDiag = new double[fSize];
//            Arrays.fill(piDiag, 0);
//
//            double[] pi = new double[fSize];
//            double[] oldPi = new double[fSize];
//
//            for(int k=0; k<fSize; k++){
//                pi[k] = Math.log(cDoc.m_sstat[k]);
//                oldPi[k] = Math.log(cDoc.m_sstat[k]);
//            }
//
//            try {
//                do {
//                    double diff = 0;
//
//                    fValue = piFuncGradientVal(pDoc, gammaSum, cDoc, pi, piGradient);
//                    LBFGS.lbfgs(fSize, 4, pi, fValue, piGradient, false, piDiag, iprint, 1e-2, 1e-10, iflag);
//
//                    for(int k=0; k<fSize; k++){
//                        double tempDiff = 0;
//                        tempDiff = pi[k]-oldPi[k];
//                        if(Math.abs(tempDiff)>diff){
//                            diff = Math.abs(tempDiff);
//                        }
//                        oldPi[k] = pi[k];
//                    }
//
//                    if(diff<m_lbfgsConverge){
////                        System.out.print("diff\t"+diff+"finish update pi");
//                        break;
//                    }
//
//
//
//                } while (iflag[0] != 0);
//            } catch (LBFGS.ExceptionWithIflag e) {
//                e.printStackTrace();
//            }
//
//            for(int k=0; k<fSize; k++){
//                cDoc.m_sstat[k] = Math.exp(pi[k]);
////                System.out.println(cDoc.getName()+"\tcDoc.m_sstat[]"+cDoc.m_sstat[k]);
//            }
//        }
//
//    }

    public double piFuncGradientVal(_ParentDoc4DCM pDoc, double gammaSum, _ChildDoc cDoc, double[]pi, double[] piGradient){
        double funcVal = 0;
        Arrays.fill(piGradient, 0);

        double expPiSum = 0;
        double[] expPi = new double[number_of_topics];
        for(int k=0; k<number_of_topics; k++){
            expPi[k] = Math.exp(pi[k]);
            expPiSum += expPi[k];
        }

        double constantGradient = 0;
        for(int k=0; k<number_of_topics; k++){
            constantGradient += (m_alpha[k]-expPi[k])*Utils.digamma(expPiSum);

//            constantGradient += (m_alpha_c[k]-expPi[k])*Utils.digamma(expPiSum);
        }

        funcVal -= Utils.lgamma(expPiSum);
        for(int k=0; k<number_of_topics; k++){
//            funcVal += (m_alpha_c[k] - 1) * (Utils.digamma(expPi[k]) - Utils.digamma(expPiSum));
            funcVal += (m_alpha[k] - 1) * (Utils.digamma(expPi[k]) - Utils.digamma(expPiSum));

            funcVal -= (expPi[k]-1)*(Utils.digamma(expPi[k])-Utils.digamma(expPiSum));
            funcVal += Utils.lgamma(expPi[k]);

            piGradient[k] = (m_alpha[k]-expPi[k])*Utils.digamma(expPi[k]);
//            piGradient[k] = (m_alpha_c[k]-expPi[k])*Utils.digamma(expPi[k]);

//            piGradient[k] -= (m_alpha_c[k]-expPi[k])*Utils.digamma(expPiSum);
        }

        _SparseFeature[] cDocFvs = cDoc.getSparse();
        for(int n=0; n<cDocFvs.length; n++) {
            int wID = cDocFvs[n].getIndex();
            double wVal = cDocFvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                funcVal += cDoc.m_phi[n][k]*wVal*(Utils.digamma(expPi[k])-Utils.digamma(expPiSum));
                funcVal -= cDoc.m_phi[n][k]*wVal*(Utils.dotProduct(expPi, pDoc.m_sstat)/(expPiSum*gammaSum*cDoc.m_zeta));

                constantGradient += cDoc.m_phi[n][k]*wVal*Utils.digamma(expPi[k])+cDoc.m_phi[n][k]*Utils.digamma(expPiSum);
//                piGradient[k] += cDoc.m_phi[n][k]*wVal*Utils.digamma(expPi[k])+cDoc.m_phi[n][k]*Utils.digamma(expPiSum);

                double temp = pDoc.m_sstat[k]*expPiSum-Utils.dotProduct(expPi, pDoc.m_sstat);

                piGradient[k] -= cDoc.m_phi[n][k]*wVal*temp/(expPiSum*expPiSum*gammaSum*cDoc.m_zeta);
            }

        }

        for(int k=0; k<number_of_topics; k++){
            piGradient[k]  -= constantGradient;
            piGradient[k] = piGradient[k]*expPi[k];
            piGradient[k] = 0-piGradient[k];
        }

        return -funcVal;
    }

    public void updateZeta4Child(_ParentDoc4DCM pDoc){
        double totalGamma4Parent = Utils.sumOfArray(pDoc.m_sstat);
        for(_ChildDoc cDoc:pDoc.m_childDocs){
            double totalPi4Child = Utils.sumOfArray(cDoc.m_sstat);
            double gammaPiInnerProd = Utils.dotProduct(pDoc.m_sstat, cDoc.m_sstat);
            cDoc.m_zeta = gammaPiInnerProd/(totalGamma4Parent*totalPi4Child);
        }
    }

    //to be improved
    double logSumOfExponentials(double[] xs){
        if(xs.length == 1){
            return xs[0];
        }

        double sum = 0;

        double max = xs[0];
        for(int i=1; i<xs.length; i++){
            if(xs[i]>max){
                max = xs[i];
            }
        }

        for (int i = 0; i < xs.length; i++) {
            if (!Double.isInfinite(xs[i])) {
                sum += Math.exp(xs[i]-max);
            }
        }

        return Math.log(sum)+max;
    }

    public void updateEta4Child(_ChildDoc cDoc){
//        for(_ChildDoc cDoc:pDoc.m_childDocs) {
            _SparseFeature[] fvs = cDoc.getSparse();
            for (int n = 0; n < fvs.length; n++) {
                int wID = fvs[n].getIndex();
                double wVal = fvs[n].getValue();

                for (int k = 0; k < number_of_topics; k++) {
                    cDoc.m_phi[n][k] = Utils.digamma(cDoc.m_sstat[k]);
                    cDoc.m_phi[n][k] +=Math.log(topic_term_probabilty[k][wID]);
                }

                double logSum = logSumOfExponentials(cDoc.m_phi[n]);

                double phiSum = 0;
                for(int k=0; k<number_of_topics; k++){
                    cDoc.m_phi[n][k] = Math.exp(cDoc.m_phi[n][k] - logSum);

//                    if((cDoc.m_phi[n][k]-logSum)<-200){
//                        cDoc.m_phi[n][k] = 1e-20;
//                    }else {
//                        cDoc.m_phi[n][k] = Math.exp(cDoc.m_phi[n][k] - logSum);
//                    }
                    phiSum += cDoc.m_phi[n][k];
                }
                if(Math.abs(phiSum-1)>1)
                    System.out.println("cDoc phisum error");
            }
//        }
    }

//    public void updateEta4Child(_ParentDoc4DCM pDoc){
//        for(_ChildDoc cDoc:pDoc.m_childDocs) {
//            _SparseFeature[] fvs = cDoc.getSparse();
//            for (int n = 0; n < fvs.length; n++) {
//                int wID = fvs[n].getIndex();
//                double wVal = fvs[n].getValue();
//
//                for (int k = 0; k < number_of_topics; k++) {
//                    cDoc.m_phi[n][k] = Utils.digamma(pDoc.m_sstat[k])+Utils.digamma(cDoc.m_sstat[k]);
//                    cDoc.m_phi[n][k] +=Math.log(topic_term_probabilty[k][wID]);
//                }
//
//                double logSum = logSumOfExponentials(cDoc.m_phi[n]);
//
//                double phiSum = 0;
//                for(int k=0; k<number_of_topics; k++){
//                    if((cDoc.m_phi[n][k]-logSum)<-200){
//                        cDoc.m_phi[n][k] = 1e-20;
//                    }else {
//                        cDoc.m_phi[n][k] = Math.exp(cDoc.m_phi[n][k] - logSum);
//                    }
//                    phiSum += cDoc.m_phi[n][k];
//                }
//            }
//        }
//    }

    public void calculate_M_step(int iter){
        updatePhi();

        if(iter%5!=4)
            return;
        updateAlpha4Parent();
//        updateAlpha4Child();
//        updateBeta();


        for(int k=0; k<number_of_topics; k++){
            System.out.println("alpha\t"+m_alpha[k]);

//            System.out.println("alpha_c\t"+m_alpha_c[k]);
//            for(int v=0; v<vocabulary_size; v++){
//                System.out.println(k+"\t"+v+"\t"+topic_term_probabilty[k][v]);
//            }
        }
    }

    public void updateAlpha4Parent(){
        updateParamViaNewtonMethod(m_alpha, m_parentDocNum+m_childDocNum, m_alpha_stat);

    }

    public void updateAlpha4Child(){
        updateParamViaNewtonMethod(m_alpha_c, m_childDocNum, m_alpha_c_stat);

    }

    public void updateBeta(){
        for(int k=0; k<number_of_topics; k++){
            updateParamViaNewtonMethod(m_beta[k], m_parentDocNum, m_beta_stat[k]);

        }
    }

    public void updatePhi(){
        double topicWordSum = 0;
        for(int k=0; k<number_of_topics; k++) {
            topicWordSum = Utils.sumOfArray(word_topic_sstat[k]);
            for (int v = 0; v < vocabulary_size; v++) {
                topic_term_probabilty[k][v] = word_topic_sstat[k][v]/topicWordSum;
            }
        }
    }

    //update parameters via newton method
    protected void updateParamViaNewtonMethod(double[] param, double paramMultiplier, double[] paramConstant){
        int iterIndex=0;
        double diff = 0;
        int paramSize = param.length;
        double[] paramUpdate = new double[param.length];
        double[] paramDiag = new double[param.length];
        double[] paramGradient = new double[param.length];

        double paramHessianConstant = 0;
        double paramSum = 0;

        do {
            paramHessianConstant = 0;
            paramSum = Utils.sumOfArray(param);
            Arrays.fill(paramUpdate, 0);
            Arrays.fill(paramDiag, 0);
            Arrays.fill(paramGradient, 0);

            for(int i=0; i<paramSize; i++){
                paramGradient[i] = paramMultiplier*(Utils.digamma(paramSum)-Utils.digamma(param[i]));
                paramGradient[i] += paramConstant[i];

                paramDiag[i] = -paramMultiplier*(Utils.trigamma(param[i]));
            }
            paramHessianConstant = paramMultiplier*Utils.trigamma(paramSum);

            double paramDiagInverseSum = 0;
            double paramDiagInverseParamGradientSum = 0;
            for (int i = 0; i < param.length; i++) {
                paramDiagInverseSum += 1 / paramDiag[i];
                paramDiagInverseParamGradientSum += paramGradient[i] / paramDiag[i];
            }
            double c = 0;
            c = (1.0 / paramHessianConstant) + paramDiagInverseSum;
            c =  paramDiagInverseParamGradientSum/c;

            diff = 0;
            for (int i = 0; i < param.length; i++) {
                paramUpdate[i] = (paramGradient[i] - c) / paramDiag[i];
                diff += paramUpdate[i]*paramUpdate[i];
                // if(Math.abs(paramUpdate[i])>diff)
                    // diff = Math.abs(paramUpdate[i]);
                param[i] -= paramUpdate[i];
            }

            diff /= number_of_topics;

            if(diff<m_varConverge)
                break;

        }while(++iterIndex<m_varMaxIter);

    }

    protected void finalEst(){
        for(_Doc d:m_trainSet){
            estThetaInDoc(d);
        }

    }

    protected void estTopicWordDistribution4Parent(_ParentDoc4DCM pDoc){
        for(int k=0; k<number_of_topics; k++){
            for(int v=0; v<vocabulary_size; v++){
                pDoc.m_wordTopic_prob[k][v] = pDoc.m_lambda_stat[k][v]/pDoc.m_lambda_topicStat[k];
            }
        }
    }

    public double calculate_log_likelihood(_Doc d){

        _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
        double logLikelihood = 0;
        double gammaSum = Utils.sumOfArray(pDoc.m_sstat);
        double alphaSum = Utils.sumOfArray(m_alpha);

        logLikelihood += Utils.lgamma(alphaSum);
        logLikelihood -= Utils.lgamma(gammaSum);
        for(int k=0; k<number_of_topics; k++){
            logLikelihood += -Utils.lgamma(m_alpha[k])+(m_alpha[k]-1)*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum));
            logLikelihood += Utils.lgamma(pDoc.m_sstat[k]);
            logLikelihood -= (pDoc.m_sstat[k]-1)*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum));
        }

        _SparseFeature[] fvs = pDoc.getSparse();
        for(int n=0; n<fvs.length; n++){
            int wID = fvs[n].getIndex();
            double wVal = fvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                double updateLikelihood = 0;

                updateLikelihood -= pDoc.m_phi[n][k]*(Math.log(pDoc.m_phi[n][k]));

                updateLikelihood += pDoc.m_phi[n][k]*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum));

                updateLikelihood += pDoc.m_phi[n][k]*wVal*(Math.log(topic_term_probabilty[k][wID]));


                logLikelihood += updateLikelihood;

            }
        }

//        double alphaCSum = Utils.sumOfArray(m_alpha_c);
        double alphaCSum = Utils.sumOfArray(m_alpha);


        for(_ChildDoc cDoc:pDoc.m_childDocs){
//            logLikelihood += Utils.lgamma(alphaCSum);
            logLikelihood += Utils.lgamma(alphaSum);

            double piSum = Utils.sumOfArray(cDoc.m_sstat);

            logLikelihood -= Utils.lgamma(piSum);

            for (int k = 0; k < number_of_topics; k++) {

//                logLikelihood -= Utils.lgamma(m_alpha_c[k]);
                logLikelihood -= Utils.lgamma(m_alpha[k]);

//                logLikelihood += (m_alpha_c[k] - 1) * (Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum));
                logLikelihood += (m_alpha[k] - 1) * (Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum));

                logLikelihood += Utils.lgamma(cDoc.m_sstat[k]);
                logLikelihood -= (cDoc.m_sstat[k]-1)*(Utils.digamma(cDoc.m_sstat[k])-Utils.digamma(piSum));
            }

            _SparseFeature[] cDocFvs = cDoc.getSparse();
            for(int n=0; n<cDocFvs.length; n++) {
                int wID = cDocFvs[n].getIndex();
                double wVal = cDocFvs[n].getValue();
                for(int k=0; k<number_of_topics; k++){
//                    logLikelihood += cDoc.m_phi[n][k]*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum)+Utils.digamma(cDoc.m_sstat[k])-Utils.digamma(piSum));
                    logLikelihood += cDoc.m_phi[n][k]*(Utils.digamma(cDoc.m_sstat[k])-Utils.digamma(piSum));
//                    logLikelihood -= cDoc.m_phi[n][k]*(Utils.dotProduct(cDoc.m_sstat, pDoc.m_sstat)/(piSum*gammaSum*cDoc.m_zeta)+Math.log(cDoc.m_zeta)-1);
                    logLikelihood += wVal*cDoc.m_phi[n][k]*(Math.log(topic_term_probabilty[k][wID]));

                    logLikelihood -= cDoc.m_phi[n][k]*Math.log(cDoc.m_phi[n][k]);

                }
            }


        }


//        System.out.println("doc \t"+pDoc.getName()+"\t likelihood \t"+logLikelihood);
        return logLikelihood;
    }

    public double calculate_log_likelihood4Parent(_ParentDoc4DCM pDoc){

        double logLikelihood = 0;
        double gammaSum = Utils.sumOfArray(pDoc.m_sstat);
        double alphaSum = Utils.sumOfArray(m_alpha);

        logLikelihood += Utils.lgamma(alphaSum);
        logLikelihood -= Utils.lgamma(gammaSum);
        for(int k=0; k<number_of_topics; k++){
            logLikelihood += -Utils.lgamma(m_alpha[k])+(m_alpha[k]-1)*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum));
            logLikelihood += Utils.lgamma(pDoc.m_sstat[k]);
            logLikelihood -= (pDoc.m_sstat[k]-1)*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum));
        }

        _SparseFeature[] fvs = pDoc.getSparse();
        for(int n=0; n<fvs.length; n++){
            int wID = fvs[n].getIndex();
            double wVal = fvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
                double updateLikelihood = 0;

                updateLikelihood -= pDoc.m_phi[n][k]*(Math.log(pDoc.m_phi[n][k]));

                updateLikelihood += pDoc.m_phi[n][k]*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum));

                updateLikelihood += pDoc.m_phi[n][k]*wVal*(Math.log(topic_term_probabilty[k][wID]));


                logLikelihood += updateLikelihood;

            }
        }
//        System.out.println("doc \t"+pDoc.getName()+"\t likelihood \t"+logLikelihood);
        return logLikelihood;
    }

    public double calculate_log_likelihood4Child(_ChildDoc cDoc){

        double logLikelihood = 0;
        double alphaSum = Utils.sumOfArray(m_alpha);

//            logLikelihood += Utils.lgamma(alphaCSum);
        logLikelihood += Utils.lgamma(alphaSum);

        double piSum = Utils.sumOfArray(cDoc.m_sstat);

        logLikelihood -= Utils.lgamma(piSum);

        for (int k = 0; k < number_of_topics; k++) {
//                logLikelihood -= Utils.lgamma(m_alpha_c[k]);
//                logLikelihood += (m_alpha_c[k] - 1) * (Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum));

            logLikelihood -= Utils.lgamma(m_alpha[k]);

            logLikelihood += (m_alpha[k] - 1) * (Utils.digamma(cDoc.m_sstat[k]) - Utils.digamma(piSum));

            logLikelihood += Utils.lgamma(cDoc.m_sstat[k]);
            logLikelihood -= (cDoc.m_sstat[k]-1)*(Utils.digamma(cDoc.m_sstat[k])-Utils.digamma(piSum));
        }

        _SparseFeature[] cDocFvs = cDoc.getSparse();
        for(int n=0; n<cDocFvs.length; n++) {
            int wID = cDocFvs[n].getIndex();
            double wVal = cDocFvs[n].getValue();
            for(int k=0; k<number_of_topics; k++){
//                    logLikelihood += cDoc.m_phi[n][k]*(Utils.digamma(pDoc.m_sstat[k])-Utils.digamma(gammaSum)+Utils.digamma(cDoc.m_sstat[k])-Utils.digamma(piSum));
                logLikelihood += cDoc.m_phi[n][k]*(Utils.digamma(cDoc.m_sstat[k])-Utils.digamma(piSum));
//                    logLikelihood -= cDoc.m_phi[n][k]*(Utils.dotProduct(cDoc.m_sstat, pDoc.m_sstat)/(piSum*gammaSum*cDoc.m_zeta)+Math.log(cDoc.m_zeta)-1);
                logLikelihood += wVal*cDoc.m_phi[n][k]*(Math.log(topic_term_probabilty[k][wID]));

                logLikelihood -= cDoc.m_phi[n][k]*Math.log(cDoc.m_phi[n][k]);

            }
        }

//        System.out.println("doc \t"+pDoc.getName()+"\t likelihood \t"+logLikelihood);
        return logLikelihood;
    }

    protected double calculate_log_likelihood() {

        double corpusLogLikelihood = 0;
        for(_Doc d:m_trainSet){
            if(d instanceof _ParentDoc4DCM){
                _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
                corpusLogLikelihood += calculate_log_likelihood(pDoc);
            }
        }

        return corpusLogLikelihood;
    }


}
