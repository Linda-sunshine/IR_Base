package Analyzer;

import opennlp.tools.util.InvalidFormatException;
import org.apache.commons.math3.distribution.BinomialDistribution;
import structures._Doc;
import structures._Review;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The analyzer aims at the directional network information.
 */
public class MultiThreadedNetworkAnalyzer extends MultiThreadedLinkPredAnalyzer {

    // Decide if the network is directional or not
    private boolean m_directionFlag = false;
    private HashMap<String, HashSet<String>> m_interactionMap = new HashMap<>();
    private HashMap<String, HashSet<String>>  m_nonInteractionMap = new HashMap<>();
    private BinomialDistribution m_bernoulli;
    private double m_rho;

    public MultiThreadedNetworkAnalyzer(String tokenModel, int classNo,
                                        String providedCV, int Ngram, int threshold, int numberOfCores, boolean b, double rho)
            throws InvalidFormatException, FileNotFoundException, IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b);
        m_rho = rho;
    }

    protected void loadRawNetwork(String filename){
        try{
            // step 1: load the raw network file
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            // load the interactions first
            while((line = reader.readLine()) != null){
                String[] users = line.trim().split("\t");
                String[] interactions = Arrays.copyOfRange(users, 1, users.length);
                m_userIDs.add(users[0]);

                if(interactions.length == 0)
                    return;
                else
                    m_interactionMap.put(users[0], new HashSet<>(Arrays.asList(interactions)));
            }
            reader.close();

        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // construct the network structure
    public void constructNetwork(String filename){

        // step 1: load the raw network file
        loadRawNetwork(filename);
        // init the bernoulli sampler
        initBernoulliSampler();

        if(m_directionFlag){
            for(String uid: m_userIDs) {
                // step 2: clean-up of the users that are not in the current user set
                if (m_interactionMap.containsKey(uid)) {
                    updateDirectionalNetworkMaps(uid);
                }
                // step 3: sample non-interactions whether user has friends or not
                sampleDirectionalNonInteractions(uid);
            }
        } else{
            // step 2: clean-up of the users that are not in the current user set
            for(String uid: m_userIDs) {
                if (m_interactionMap.containsKey(uid)) {
                    updateNonDirectionalInteractions(uid);
                }
            }
            // step 3: sample non-interactions whether user has friends or not
            // the non-interactions is reflected in the global map directly.
            for(String uid: m_userIDs){
                sampleNonDirectionalNonInteractions(uid);
            }
        }
    }

    protected void initBernoulliSampler(){
        m_bernoulli = new BinomialDistribution(1, m_rho);
    }

    // update m_interactionMap and m_nonInteractionMap
    protected void updateDirectionalNetworkMaps(String uid){
        HashSet<String> rawInteractions = m_interactionMap.get(uid);
        HashSet<String> realInteractions = new HashSet<>();
        HashSet<String> nonInteractions = new HashSet<>(m_userIDs);
        nonInteractions.remove(uid);

        for(String intId: rawInteractions){
            if(m_userIDs.contains(intId)) {
                realInteractions.add(intId);
                // collect complete set of non-interactions
                nonInteractions.remove(intId);
            }
        }
        if(realInteractions.size() == 0)
            m_interactionMap.remove(uid);
        else
            m_interactionMap.put(uid, realInteractions);

        // store the complete set of non-interactions for each user first
        m_nonInteractionMap.put(uid, nonInteractions);
    }


    // update m_interactionMap and m_nonInteractionMap
    protected void updateNonDirectionalInteractions(String uid){
        HashSet<String> rawInteractions = m_interactionMap.get(uid);
        HashSet<String> realInteractions = new HashSet<>();

        // symmteric interactions
        for(String intId: rawInteractions){
            if(m_userIDs.contains(intId)) {
                realInteractions.add(intId);
                m_interactionMap.get(intId).add(uid);
            }
        }
        if(realInteractions.size() == 0)
            m_interactionMap.remove(uid);
        else
            m_interactionMap.put(uid, realInteractions);
    }

    protected void sampleDirectionalNonInteractions(String uid){
        HashSet<String> sampledNonInteractions = new HashSet<>();
        for(String nonInt: m_nonInteractionMap.get(uid)){
            if(m_bernoulli.sample() == 1)
                sampledNonInteractions.add(nonInt);
        }
        m_nonInteractionMap.put(uid, sampledNonInteractions);
    }

    protected void sampleNonDirectionalNonInteractions(String uid){
        // sample non-interactions
        HashSet<String> nonInteractions = new HashSet<String>(m_userIDs);
        nonInteractions.remove(uid);

        if(m_interactionMap.containsKey(uid)){
            for(String intId: m_interactionMap.get(uid)){
                nonInteractions.remove(intId);
            }
        }

        HashSet<String> sampledNonInteractions = new HashSet<>();
        for(String nonInt: nonInteractions){
            if(m_bernoulli.sample() == 1) {
                sampledNonInteractions.add(nonInt);
                if(!m_nonInteractionMap.containsKey(nonInt))
                    m_nonInteractionMap.put(nonInt, new HashSet<>());
                m_nonInteractionMap.get(nonInt).add(uid);
            }
        }
        if(!m_nonInteractionMap.containsKey(uid))
            m_nonInteractionMap.put(uid, new HashSet<>());
        m_nonInteractionMap.getOrDefault(uid, new HashSet<>()).addAll(sampledNonInteractions);
    }
    // save the network for later use
    public void saveNetwork(String filename, HashMap<String, HashSet<String>> map){
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: map.keySet()){
                writer.write(uid + '\t');
                for(String it: map.get(uid)){
                    writer.write(it + '\t');
                }
                writer.write('\n');
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }


    public HashMap<String, HashSet<String>> getInteractionMap(){
        return m_interactionMap;
    }

    public HashMap<String, HashSet<String>> getNonInteractionMap(){
        return m_nonInteractionMap;
    }

    public void loadInteractions(String filename){
        try{
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            // load the interactions first
            while((line = reader.readLine()) != null){
                String[] users = line.trim().split("\t");
                String[] interactions = Arrays.copyOfRange(users, 1, users.length);
                if(!m_userIDIndex.containsKey(users[0]))
                    System.err.println("[error] The user does not exist: " + users[0]);
                else {
                    int uIndex = m_userIDIndex.get(users[0]);
                    m_users.get(uIndex).setFriends(interactions);
                }
            }
            reader.close();

        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void loadNonInteractions(String filename){
        try{
            // step 1: load the raw network file
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            // load the interactions first
            while((line = reader.readLine()) != null){
                String[] users = line.trim().split("\t");
                String[] nonInteractions = Arrays.copyOfRange(users, 1, users.length);
                if(!m_userIDIndex.containsKey(users[0]))
                    System.err.println("[error] The user does not exist: " + users[0]);
                else {
                    int uIndex = m_userIDIndex.get(users[0]);
                    m_users.get(uIndex).setNonFriends(nonInteractions);
                }
            }
            reader.close();

        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // shuffle the whole corpus and save the index information for later use
    public void saveCVIndex(int k, String filename){
        m_corpus.setMasks();
        m_corpus.shuffle(k);
        int[] masks = m_corpus.getMasks();
        ArrayList<_Doc> docs = m_corpus.getCollection();
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int i=0; i<docs.size(); i++){
                _Review r = (_Review) docs.get(i);
                writer.write(String.format("%s,%d,%d\n", r.getUserID(), r.getID(), masks[i]));
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    @Override
    public void constructUserIDIndex(){
        m_userIDIndex = new HashMap<String, Integer>();
        for(int i=0; i<m_users.size(); i++){
            m_userIDIndex.put(m_users.get(i).getUserID(), i);
        }
    }

    // load cv index for all the documents
    public void loadCVIndex(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while((line = reader.readLine()) != null) {
                String[] strs = line.trim().split(",");
                String uId = strs[0];
                int id = Integer.valueOf(strs[1]);
                int mask = Integer.valueOf(strs[2]);

                if(!m_userIDIndex.containsKey(uId))
                    System.out.println("No such user!");
                else {
                    int uIndex = m_userIDIndex.get(uId);
                    if (uIndex > m_users.size())
                        System.out.println("Exceeds the array size!");
                    else {
                        m_users.get(m_userIDIndex.get(uId)).getReviewByID(id).setMask4CV(mask);
                    }
                }
            }
        } catch(IOException e){
            e.printStackTrace();
        }
    }

}
