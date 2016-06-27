package topicmodels;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import structures._ChildDoc;
import structures._ChildDoc4TwoPhi;
import structures._Corpus;
import structures._ChildDoc4TwoPhi;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4ThreePhi;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ParentChildWith2Phi extends ParentChild_Gibbs {

	public double[] m_childTopicProbCache;
	public double[] m_gammaParent; // 2 dimensions in parent
	public double[] m_gammaChild; // 2 dimensions in child

	public ParentChildWith2Phi(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gammaParent, double[] gammaChild,
			double ksi, double tau) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gammaParent, ksi, tau);
		// TODO Auto-generated constructor stub

		m_topicProbCache = new double[number_of_topics + 1];
		m_childTopicProbCache = new double[number_of_topics + 1];

		m_gammaParent = new double[gammaParent.length];
		m_gammaChild = new double[gammaChild.length];

		System.arraycopy(gammaParent, 0, m_gammaParent, 0, m_gammaParent.length);
		System.arraycopy(gammaChild, 0, m_gammaChild, 0, m_gammaChild.length);

	}

	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		for (int i = 0; i < number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta * vocabulary_size); // avoid adding such
		// prior later on

		for (_Doc d : collection) {
			if (d instanceof _ParentDoc) {
				((_ParentDoc4ThreePhi) d).createXSpace(number_of_topics, m_gammaParent.length, vocabulary_size, d_beta);
				((_ParentDoc4ThreePhi) d).setTopics4Gibbs(number_of_topics, 0);
				for (_Stn stnObj : d.getSentences())
					stnObj.setTopicsVct4ThreePhi(number_of_topics, m_gammaParent.length);
			} else if (d instanceof _ChildDoc) {
				((_ChildDoc4TwoPhi) d).createXSpace(number_of_topics, m_gammaChild.length);
				((_ChildDoc4TwoPhi) d).setTopics4Gibbs(number_of_topics, 0);
				computeMu4Doc((_ChildDoc)d);
			}

		}

		for (_Doc d : collection) {
			if (d instanceof _ParentDoc4ThreePhi) {
				for (_Word w : d.getWords()) {
					int xid = w.getX();
					int tid = w.getTopic();
					int wid = w.getIndex();
					// update global
					if (xid == 0) {
						word_topic_sstat[tid][wid]++;
						m_sstat[tid]++;
					} 

				}
			} else if(d instanceof _ChildDoc4TwoPhi){
				for(_Word w: d.getWords()){
					int xid = w.getX();
					int tid = w.getTopic();
					int wid = w.getIndex();
					//update global
					if(xid==0){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
					}else if(xid==1){
					//update pair

						_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi) ((_ChildDoc4TwoPhi) d).m_parentDoc;
						pDoc.m_pairWordSstat[wid]++;
						pDoc.m_pairWord++;
					}

				}
			}
		}

		imposePrior();

		m_statisticsNormalized = false;
	}

	public void sampleInParentDoc(_ParentDoc d) {
		_ParentDoc4ThreePhi doc = (_ParentDoc4ThreePhi) d;
		int wid, tid, xid;
		double normalizedProb;
		double supplementProb;

		for (_Word w : d.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();

			if (xid == 0) {
				doc.m_xTopicSstat[xid][tid] --;
				doc.m_xSstat[xid]--;

				if (m_collectCorpusStats) {
					word_topic_sstat[tid][wid]--;
					m_sstat[tid]--;
				}
			} else if (xid == 1) {
				doc.m_pairWord --;
				doc.m_pairWordSstat[wid] --;
				
				doc.m_xTopicSstat[xid][0] --;
				doc.m_xSstat[xid] --;
			}

			normalizedProb = 0;
			supplementProb = 0;

			double pLambdaZero = parentXInDocProb(0, doc);
			double pLambdaOne = parentXInDocProb(1, doc);

			double pWordTopic = 0;
			for (tid = 0; tid < number_of_topics; tid++) {
				pWordTopic = parentWordByTopicProb(tid, wid);
				double pTopicPdoc = parentTopicInDocProb(tid, doc);
				double pTopicCdoc = parentChildInfluenceProb(tid, doc);
//				System.out.println("influence from child\t"+pTopicCdoc);
				m_topicProbCache[tid] = pWordTopic * pTopicPdoc * pTopicCdoc * pLambdaZero;
				supplementProb += pTopicPdoc * pTopicCdoc;
			}

			for (tid = 0; tid < number_of_topics; tid++) {
				m_topicProbCache[tid] /= supplementProb;
				normalizedProb += m_topicProbCache[tid];
			}

			pWordTopic = localParentWordByTopicProb(wid, doc);
			// extra one dimension
			m_topicProbCache[tid] = pWordTopic * pLambdaOne;

			normalizedProb += m_topicProbCache[tid];

			normalizedProb *= m_rand.nextDouble();
			for (tid = 0; tid < m_topicProbCache.length; tid++) {
				normalizedProb -= m_topicProbCache[tid];
				if (normalizedProb <= 0)
					break;
			}

			if (tid == m_topicProbCache.length)
				tid--;

			if (tid < number_of_topics) {

				xid = 0;
				w.setX(xid);
				w.setTopic(tid);
				doc.m_xTopicSstat[xid][tid]++;
				doc.m_xSstat[xid]++;
				
				if (m_collectCorpusStats) {
					word_topic_sstat[tid][wid]++;
					m_sstat[tid]++;
				}
			} else {
				xid = 1;
				tid = number_of_topics;
				w.setX(xid);
				w.setTopic(tid);
				doc.m_xTopicSstat[xid][0]++;
				doc.m_xSstat[xid]++;
				
				doc.m_pairWordSstat[wid]++;
				doc.m_pairWord++;
			}

		}
	}

	protected double parentTopicInDocProb(int tid, _ParentDoc4ThreePhi d) {
		return d_alpha + d.m_xTopicSstat[0][tid];
	}

	// localword initialized vocabulary*beta;
	protected double localParentWordByTopicProb(int wid, _ParentDoc4ThreePhi d) {
		return (d.m_pairWordSstat[wid])
				/ (d.m_pairWord);
	}

	protected double parentXInDocProb(int xid, _ParentDoc4ThreePhi d) {
		return m_gammaParent[xid]+d.m_xSstat[xid];
	}

	protected double parentChildInfluenceProb(int tid, _ParentDoc4ThreePhi d){
		double term = 1.0;
		
		if(tid==0)
			return term;
		
		for (_ChildDoc cDoc : d.m_childDocs) {
			double muDp = cDoc.getMu() / (d.m_xSstat[0] + 1);
			term *= gammaFuncRatio((int)cDoc.m_xTopicSstat[0][tid], muDp,
					d_alpha
					+ d.m_xTopicSstat[0][tid] * muDp)
					/ gammaFuncRatio((int)cDoc.m_xTopicSstat[0][0], muDp,
							d_alpha
							+ d.m_xTopicSstat[0][0] * muDp);
		} 

		return term;
	}

	protected void sampleInChildDoc(_ChildDoc d) {
		_ChildDoc4TwoPhi doc = (_ChildDoc4TwoPhi) d;
		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi) doc.m_parentDoc;
		int wid, tid, xid;

		double normalizedProb;
		for (_Word w : d.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();

			if (xid == 0) {
				doc.m_xTopicSstat[xid][tid]--;
				doc.m_xSstat[xid]--;

				if (m_collectCorpusStats) {
					word_topic_sstat[tid][wid]--;
					m_sstat[tid]--;
				}
			} else if (xid == 1) {
				doc.m_xTopicSstat[xid][0]--;
				doc.m_xSstat[xid]--;
				
				pDoc.m_pairWordSstat[wid]--;
				pDoc.m_pairWord --;
			}

			normalizedProb = 0;
			double pLambdaZero = childXInDocProb(0, doc);
			double pLambdaOne = childXInDocProb(1, doc);

			double pWordTopic = 0;
			for (tid = 0; tid < number_of_topics; tid++) {
				pWordTopic = childWordByTopicProb(tid, wid);

				double pTopic = childTopicInDocProb(tid, doc, pDoc);

				m_childTopicProbCache[tid] = pWordTopic * pTopic * pLambdaZero;
				normalizedProb += m_childTopicProbCache[tid];
			}

			pWordTopic = childParentWordByTopicProb(wid, pDoc);
			m_childTopicProbCache[tid] = pWordTopic * pLambdaOne;
			normalizedProb += m_childTopicProbCache[tid];

			normalizedProb *= m_rand.nextDouble();
			for (tid = 0; tid < m_childTopicProbCache.length; tid++) {
				normalizedProb -= m_childTopicProbCache[tid];
				if (normalizedProb <= 0)
					break;
			}

			if (tid == m_childTopicProbCache.length)
				tid--;

			if (tid < number_of_topics) {
				xid = 0;
				w.setX(xid);
				w.setTopic(tid);
				
				doc.m_xTopicSstat[xid][tid]++;
				doc.m_xSstat[xid]++;

				if (m_collectCorpusStats) {
					word_topic_sstat[tid][wid]++;
					m_sstat[tid]++;
				}

			} else if (tid == number_of_topics) {
				xid = 1;
				w.setX(xid);
				w.setTopic(tid);
				doc.m_xTopicSstat[xid][0]++;
				doc.m_xSstat[xid]++;
			
				pDoc.m_pairWordSstat[wid]++;
				pDoc.m_pairWord ++;
			}

		}
	}

	protected double childXInDocProb(int xid, _ChildDoc4TwoPhi d) {
		return m_gammaChild[xid] + d.m_xSstat[xid];

	}

	protected double childParentWordByTopicProb(int wid, _ParentDoc4ThreePhi pDoc) {
		return (pDoc.m_pairWordSstat[wid])
				/ (pDoc.m_pairWord);
	}

	protected double childTopicInDocProb(int tid, _ChildDoc4TwoPhi cDoc, _ParentDoc4ThreePhi pDoc){
		double docLength = pDoc.m_xSstat[0];
		
		return (d_alpha + cDoc.getMu() * pDoc.m_xTopicSstat[0][tid]
					/ docLength + cDoc.m_xTopicSstat[0][tid])
				/ (m_kAlpha + cDoc.getMu() + cDoc.m_xSstat[0]);	
	}
	
	protected void estThetaInDoc(_Doc d) {
		if (d instanceof _ParentDoc4ThreePhi) {
			// estimate topic proportion of sentences in parent documents
			((_ParentDoc4ThreePhi) d).estStnTheta();
			((_ParentDoc4ThreePhi) d).estGlobalLocalTheta();
		} else if (d instanceof _ChildDoc4TwoPhi) {
			((_ChildDoc4TwoPhi) d).estGlobalLocalTheta();
		}
		m_statisticsNormalized = true;
	}

	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d) {
		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi) d;

		pDoc.createXSpace(number_of_topics, m_gammaParent.length, vocabulary_size, d_beta);
		pDoc.setTopics4Gibbs(number_of_topics, 0);
		for (_Stn stnObj : pDoc.getSentences())
			stnObj.setTopicsVct4ThreePhi(number_of_topics, m_gammaParent.length);
		sampleTestSet.add(pDoc);
	
		
		for (_ChildDoc cDoc : pDoc.m_childDocs) {
			_ChildDoc4TwoPhi childDoc = (_ChildDoc4TwoPhi) cDoc;
			childDoc.createXSpace(number_of_topics, m_gammaChild.length);		
			computeMu4Doc(childDoc);

			childDoc.setTopics4Gibbs(number_of_topics, 0);
			sampleTestSet.add(childDoc);

			for (_Word w : childDoc.getWords()) {
				int xid = w.getX();
				int tid = w.getTopic();
				int wid = w.getIndex();

				if (xid == 1) {
					// update pair
					pDoc.m_pairWordSstat[wid]++;
					pDoc.m_pairWord ++;
				}
			}
		}
	}


	protected double logLikelihoodByIntegrateTopics(_ParentDoc d) {
		_ParentDoc4ThreePhi doc = (_ParentDoc4ThreePhi) d;
		double docLogLikelihood = 0.0;
		_SparseFeature[] fv = d.getSparse();
		double gammaLen = Utils.sumOfArray(m_gammaParent);

		for (int j = 0; j < fv.length; j++) {
			int wid = fv[j].getIndex();
			double value = fv[j].getValue();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)
						*parentTopicInDocProb(k, doc)/(doc.m_xSstat[0]+number_of_topics*d_alpha)
						* parentXInDocProb(0, doc)
						/ (doc.getTotalDocLength() + gammaLen);
				wordLogLikelihood += wordPerTopicLikelihood;

			}

			wordLogLikelihood += localParentWordByTopicProb(wid, doc)
					* parentXInDocProb(1, doc)
					/ (doc.getTotalDocLength() + gammaLen);

			if (Math.abs(wordLogLikelihood) < 1e-10) {
				System.out.println("wordLoglikelihood\t" + wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);
			docLogLikelihood += value * wordLogLikelihood;
		}

		return docLogLikelihood;
	}

	protected double logLikelihoodByIntegrateTopics(_ChildDoc d) {
		_ChildDoc4TwoPhi doc = (_ChildDoc4TwoPhi) d;
		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi) doc.m_parentDoc;
		double docLogLikelihood = 0.0;

		// prepare compute the normalizers
		_SparseFeature[] fv = d.getSparse();
		double gammaLen = Utils.sumOfArray(m_gammaChild);
		
		for (int i=0; i<fv.length; i++) {
			int wid = fv[i].getIndex();
			double value = fv[i].getValue();

			double wordLogLikelihood = 0;
			
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
						* childTopicInDocProb(k, doc, pDoc)
						* childXInDocProb(0, doc)
						/ (doc.getTotalDocLength() + gammaLen);

				wordLogLikelihood += wordPerTopicLikelihood;
				
				if(Double.isNaN(wordPerTopicLikelihood)){
					System.out.println("nan word likelihood in child Doc");
				}
			}

			wordLogLikelihood += childParentWordByTopicProb(wid, pDoc)
					* childXInDocProb(1, doc)
					/ (doc.getTotalDocLength() + gammaLen);
			

			if(Math.abs(wordLogLikelihood) < 1e-10){
				System.out.println("wordLoglikelihood\t"+wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);
	
			docLogLikelihood += value * wordLogLikelihood;
		}
		
		return docLogLikelihood;
	}
	

	public void collectParentStats(_ParentDoc d) {
		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi) d;
		for (int k = 0; k < this.number_of_topics; k++) {
			pDoc.m_topics[k] += pDoc.m_xTopicSstat[0][k] + d_alpha;
		}
		pDoc.m_topics[number_of_topics] += pDoc.m_xTopicSstat[1][0];

		for(int x=0; x<m_gammaParent.length; x++)
			pDoc.m_xProportion[x] += m_gammaParent[x] + pDoc.m_xSstat[x];
		
		for(int w=0; w<vocabulary_size; w++)
			pDoc.m_pairWordDistribution[w] += pDoc.m_pairWordSstat[w];
		
		pDoc.collectTopicWordStat();
		
		for(_Word w:d.getWords()){
			w.collectXStats();
		}
	}

	public void collectChildStats(_ChildDoc d) {
		_ChildDoc4TwoPhi cDoc = (_ChildDoc4TwoPhi)d;
		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi)cDoc.m_parentDoc;

		double parentDocLength = pDoc.m_xSstat[0];
		double temp = 0;

		for(int k=0; k<this.number_of_topics; k++){
			temp = cDoc.m_xTopicSstat[0][k] + d_alpha
					+ cDoc.getMu()*pDoc.m_xTopicSstat[0][k] / parentDocLength;

			cDoc.m_topics[k] += temp;
			cDoc.m_xTopics[0][k] += temp;

		}
		cDoc.m_topics[number_of_topics] += cDoc.m_xTopicSstat[1][0];
		cDoc.m_xTopics[1][0] += cDoc.m_xTopicSstat[1][0];

		cDoc.m_topics[number_of_topics] += cDoc.m_xTopicSstat[1][0];

		for (int x = 0; x < m_gammaChild.length; x++)
			cDoc.m_xProportion[x] += m_gammaChild[x] + cDoc.m_xSstat[x];
		
		for(_Word w:d.getWords()){
			w.collectXStats();
		}

	}

	public void estParentStnTopicProportion(_ParentDoc pDoc) {
		for (_Stn stnObj : pDoc.getSentences()) {
			estStn(stnObj, (_ParentDoc4ThreePhi) pDoc);
		}
	}

	public void estStn(_Stn stnObj, _ParentDoc4ThreePhi d) {
		int i = 0;
		initStn(stnObj);
		do {
			calculateStn_E_step(stnObj, d);
			if (i > m_burnIn && i % m_lag == 0) {
				collectStnStats(stnObj, d);
			}

		} while (++i < number_of_iteration);

		Utils.L1Normalization(stnObj.m_topics);
	}

	public void initStn(_Stn stnObj) {
		stnObj.setTopicsVct4ThreePhi(number_of_topics, m_gammaParent.length);
	}

	public void calculateStn_E_step(_Stn stnObj, _ParentDoc4ThreePhi d) {
		stnObj.permuteStn();

		double normalizedProb = 0;
		int wid, tid, xid;
		for (_Word w : stnObj.getWords()) {
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();

			if (xid == 0) {
				stnObj.m_xSstat[xid]--;
				stnObj.m_topicSstat[tid]--;
			} else if (xid == 1) {
				stnObj.m_xSstat[xid]--;
				stnObj.m_topicSstat[tid]--;
			}

			normalizedProb = 0;

			double pLambdaZero = parentXInStnProb(0, stnObj, d);
			double pLambdaOne = parentXInStnProb(1, stnObj, d);

			double pWordTopic = 0;

			for (tid = 0; tid < number_of_topics; tid++) {
				pWordTopic = parentWordByTopicProb(tid, wid);
				double pTopic = parentTopicInStnProb(tid, stnObj, d);

				m_topicProbCache[tid] = pWordTopic * pTopic * pLambdaZero;
				normalizedProb += m_topicProbCache[tid];
			}

			pWordTopic = localParentWordByTopicProb(wid, d);
			m_topicProbCache[tid] = pWordTopic * pLambdaOne;
			normalizedProb += m_topicProbCache[tid];

			normalizedProb *= m_rand.nextDouble();
			for (tid = 0; tid < m_topicProbCache.length; tid++) {
				normalizedProb -= m_topicProbCache[tid];
				if (normalizedProb <= 0)
					break;
			}

			if (tid == m_topicProbCache.length)
				tid--;

			if (tid < number_of_topics) {
				xid = 0;
				w.setX(xid);
				w.setTopic(tid);
				stnObj.m_topicSstat[tid]++;
				stnObj.m_xSstat[xid]++;
			} else {
				xid = 1;
				w.setX(xid);
				w.setTopic(tid);
				stnObj.m_topicSstat[tid]++;
				stnObj.m_xSstat[xid]++;
			}
		}

	}

	public double parentXInStnProb(int xid, _Stn stnObj, _ParentDoc4ThreePhi d) {
		return m_gammaParent[xid] + (d.m_xSstat[xid] / d.getTotalDocLength())
				+ stnObj.m_xSstat[xid];
		
	}

	public double parentTopicInStnProb(int tid, _Stn stnObj, _ParentDoc4ThreePhi d) {
		return (d_alpha + d.m_xTopicSstat[0][tid] / d.m_xSstat[0] + stnObj.m_topicSstat[tid])
				/ (m_kAlpha + 1 + stnObj.m_xSstat[0]);
	}

	public void collectStnStats(_Stn stnObj, _ParentDoc4ThreePhi d) {
		for (int k = 0; k < number_of_topics; k++) {
			stnObj.m_topics[k] += stnObj.m_topicSstat[k] + d_alpha + d.m_xTopicSstat[0][k] / d.m_xSstat[0];
		}
		stnObj.m_topics[number_of_topics] += stnObj.m_topicSstat[number_of_topics];
	}

	public void debugOutput(String filePrefix) {

		File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
		File parentPairTopicDistriFolder = new File(filePrefix + "pairTopic");
		File childTopicFolder = new File(filePrefix + "childTopicAssignment");
		File childLocalWordTopicFolder = new File(filePrefix + "childLocalTopic");

		if (!parentTopicFolder.exists()) {
			System.out.println("creating directory" + parentTopicFolder);
			parentTopicFolder.mkdir();
		}
		if (!parentPairTopicDistriFolder.exists()) {
			System.out.println("creating pair directory" + parentPairTopicDistriFolder);
			parentPairTopicDistriFolder.mkdir();
		}

		if (!childTopicFolder.exists()) {
			System.out.println("creating directory" + childTopicFolder);
			childTopicFolder.mkdir();
		}
		if (!childLocalWordTopicFolder.exists()) {
			System.out.println("creating directory" + childLocalWordTopicFolder);
			childLocalWordTopicFolder.mkdir();
		}

		File parentPhiFolder = new File(filePrefix + "parentPhi");
		File childPhiFolder = new File(filePrefix + "childPhi");
		if (!parentPhiFolder.exists()) {
			System.out.println("creating directory" + parentPhiFolder);
			parentPhiFolder.mkdir();
		}
		if (!childPhiFolder.exists()) {
			System.out.println("creating directory" + childPhiFolder);
			childPhiFolder.mkdir();
		}

		File childXFolder = new File(filePrefix + "ChildXValue");
		if (!childXFolder.exists()) {
			System.out.println("creating x child Value directory" + childXFolder);
			childXFolder.mkdir();
		}
		
		File parentXFolder = new File(filePrefix + "ParentXValue");
		if (!parentXFolder.exists()) {
			System.out.println("creating x parent Value directory" + parentXFolder);
			parentXFolder.mkdir();
		}

		for (_Doc d : m_corpus.getCollection()) {
			if (d instanceof _ParentDoc) {
				printParentPhi((_ParentDoc) d, parentPhiFolder);
				printPairWordTopicDistribution((_ParentDoc4ThreePhi) d, parentPairTopicDistriFolder);
				printXValue(d, parentXFolder);
			} else if (d instanceof _ChildDoc) {
				printChildTopicAssignment(d, childTopicFolder);
				printXValue(d, childXFolder);
			}

		}

		String parentParameterFile = filePrefix + "parentParameter.txt";
		String childParameterFile = filePrefix + "childParameter.txt";
		printParameter(parentParameterFile, childParameterFile);

		String similarityFile = filePrefix + "topicSimilarity.txt";
		discoverSpecificComments(MatchPair.MP_ChildDoc, similarityFile);

		printEntropy(filePrefix);

		int topKStn = 10;
		int topKChild = 10;

		printTopKChild4Stn(filePrefix, topKChild);

		printTopKStn4Child(filePrefix, topKStn);

		printTopKChild4Parent(filePrefix, topKChild);
	}

	public void printParentPhi(_ParentDoc d, File phiFolder) {
		String parentPhiFileName = d.getName() + ".txt";
		_SparseFeature[] fv = d.getSparse();

		try {
			PrintWriter parentPW = new PrintWriter(new File(phiFolder, parentPhiFileName));

			for (int n = 0; n < fv.length; n++) {
				int index = fv[n].getIndex();
				String featureName = m_corpus.getFeature(index);
				parentPW.print(featureName + ":\t");
				for (int k = 0; k < d.m_topics.length; k++)
					parentPW.print(d.m_phi[n][k] + "\t");
				parentPW.println();
			}
			parentPW.flush();
			parentPW.close();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	public void printParameter(String parentParameterFile, String childParameterFile) {
		System.out.println("printing parameter");
		try {
			System.out.println(parentParameterFile);
			System.out.println(childParameterFile);

			PrintWriter parentParaOut = new PrintWriter(new File(parentParameterFile));
			PrintWriter childParaOut = new PrintWriter(new File(childParameterFile));
			for (_Doc d : m_corpus.getCollection()) {
				if (d instanceof _ParentDoc) {
					parentParaOut.print(d.getName() + "\t");
					parentParaOut.print("topicProportion\t");
					for (int k = 0; k < d.m_topics.length; k++) {
						parentParaOut.print(d.m_topics[k] + "\t");
					}

					for (_Stn stnObj : d.getSentences()) {
						parentParaOut.print("sentence" + (stnObj.getIndex() + 1) + "\t");
						for (int k = 0; k < d.m_topics.length; k++) {
							parentParaOut.print(stnObj.m_topics[k] + "\t");
						}
					}

					parentParaOut.println();

				} else {
					if (d instanceof _ChildDoc) {
						childParaOut.print(d.getName() + "\t");

						childParaOut.print("topicProportion\t");
						for (int k = 0; k < d.m_topics.length; k++) {
							childParaOut.print(d.m_topics[k] + "\t");
						}

						childParaOut.println();
					}
				}
			}

			parentParaOut.flush();
			parentParaOut.close();

			childParaOut.flush();
			childParaOut.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void printPairWordTopicDistribution(_ParentDoc4ThreePhi d, File parentPairTopicDistriFolder) {
		String parentLocalTopicDistriFile = d.getName() + ".txt";
		try {
			PrintWriter parentOut = new PrintWriter(new File(parentPairTopicDistriFolder, parentLocalTopicDistriFile));

			for (int wid = 0; wid < this.vocabulary_size; wid++) {
				String featureName = m_corpus.getFeature(wid);
				double wordTopicProb = d.m_pairWordDistribution[wid];
				if (wordTopicProb > 0.001)
					parentOut.format("%s:%.3f\t", featureName, wordTopicProb);
			}
			parentOut.flush();
			parentOut.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc d) {

		_ParentDoc4ThreePhi pDoc = (_ParentDoc4ThreePhi) d;
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		double gammaLen = Utils.sumOfArray(m_gammaChild);

		for (_ChildDoc cDoc : pDoc.m_childDocs) {
			int cDocLen = cDoc.getTotalDocLength();

			double stnLogLikelihood = 0;
			for (_Word w : stnObj.getWords()) {
				int wid = w.getIndex();

				double wordLogLikelihood = 0;

				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
							* childTopicInDocProb(k, (_ChildDoc4TwoPhi) cDoc, (_ParentDoc4ThreePhi) cDoc.m_parentDoc)
							* childXInDocProb(0, (_ChildDoc4TwoPhi) cDoc) / (cDoc.getTotalDocLength() + gammaLen);
					wordLogLikelihood += wordPerTopicLikelihood;
				}

				wordLogLikelihood += childParentWordByTopicProb(wid, (_ParentDoc4ThreePhi) cDoc.m_parentDoc)
						* childXInDocProb(1, (_ChildDoc4TwoPhi) cDoc) / (cDoc.getTotalDocLength() + gammaLen);

				stnLogLikelihood += Math.log(wordLogLikelihood);
			}
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}

		return childLikelihoodMap;

	}

	void discoverSpecificComments(MatchPair matchType, String similarityFile) {
		System.out.println("topic similarity");

		try {
			PrintWriter pw = new PrintWriter(new File(similarityFile));

			for (_Doc doc : m_trainSet) {
				if (doc instanceof _ParentDoc) {
					pw.print(doc.getName() + "\t");
					double stnTopicSimilarity = 0.0;
					double docTopicSimilarity = 0.0;
					for (_ChildDoc cDoc : ((_ParentDoc) doc).m_childDocs) {
						pw.print(cDoc.getName() + ":");

						docTopicSimilarity = computeSimilarity(((_ParentDoc) doc).m_topics, cDoc.m_topics);
						pw.print(docTopicSimilarity);
						for (_Stn stnObj : doc.getSentences()) {
							if (matchType == MatchPair.MP_ChildDoc)
								stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_topics);
							else if (matchType == MatchPair.MP_ChildGlobal)
								stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_xTopics[0]);
							else if (matchType == MatchPair.MP_ChildLocal)
								stnTopicSimilarity = computeSimilarity(stnObj.m_topics, cDoc.m_xTopics[1]);

							pw.print(":" + (stnObj.getIndex() + 1) + ":" + stnTopicSimilarity);
						}
						pw.print("\t");
					}
					pw.println();
				}
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

}
