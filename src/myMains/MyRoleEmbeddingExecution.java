package myMains;

import opennlp.tools.util.InvalidFormatException;
import structures.RoleParameter;
import topicmodels.RoleEmbedding.RoleEmbeddingBaseline;
import topicmodels.RoleEmbedding.RoleEmbeddingSkipGram;
import topicmodels.RoleEmbedding.UserEmbeddingBaseline;
import topicmodels.RoleEmbedding.UserEmbeddingSkipGram;

import java.io.FileNotFoundException;
import java.io.IOException;

public class MyRoleEmbeddingExecution {


    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {
        RoleParameter param = new RoleParameter(args);

        String userFile = String.format("%s/RoleEmbedding/%sUserIds.txt", param.m_prefix, param.m_data);
        String oneEdgeFile = String.format("%s/RoleEmbedding/%sCVIndex4Interaction_fold_%d_train.txt", param.m_prefix, param.m_data, param.m_fold);
        String zeroEdgeFile = String.format("%s/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_train_2.txt", param.m_prefix, param.m_data, param.m_fold);
        String baseEmbeddingFile = String.format("%s/UserEmbedding%d/%s_%s_embedding_order_%d_dim_%d_fold_%d.txt", param.m_prefix, param.m_order, param.m_data, param.m_model, param.m_order, param.m_dim, param.m_fold);
        String userEmbeddingFile = String.format("%s/UserEmbedding%d/%s_%s_embedding_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", param.m_prefix, param.m_order, param.m_data, param.m_model, param.m_order, param.m_nuOfRoles, param.m_dim, param.m_fold);
        String userEmbeddingInputFile = String.format("%s/UserEmbedding%d/%s_%s_input_embedding_order_%d_dim_%d_fold_%d.txt", param.m_prefix, param.m_order, param.m_data, param.m_model, param.m_order, param.m_dim, param.m_fold);
        String userEmbeddingOutputFile = String.format("%s/UserEmbedding%d/%s_%s_output_embedding_order_%d_dim_%d_fold_%d.txt", param.m_prefix, param.m_order, param.m_data, param.m_model, param.m_order, param.m_dim, param.m_fold);
        String roleEmbeddingFile = String.format("%s/UserEmbedding%d/%s_role_embedding_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", param.m_prefix, param.m_order, param.m_data, param.m_order, param.m_nuOfRoles, param.m_dim, param.m_fold);
        String roleTargetEmbeddingFile = String.format("%s/UserEmbedding%d/%s_role_target_embedding_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", param.m_prefix, param.m_order, param.m_data, param.m_order, param.m_nuOfRoles, param.m_dim, param.m_fold);
        String roleSourceEmbeddingFile = String.format("%s/UserEmbedding%d/%s_role_source_embedding_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", param.m_prefix, param.m_order, param.m_data, param.m_order, param.m_nuOfRoles, param.m_dim, param.m_fold);

        UserEmbeddingBaseline model = null;

        if (param.m_model.equals("user")) {
            model = new UserEmbeddingBaseline(param.m_dim, param.m_iter, param.m_converge, param.m_alpha, param.m_stepSize);
        } else if (param.m_model.equals("user_skipgram")) {
            model = new UserEmbeddingSkipGram(param.m_dim, param.m_iter, param.m_converge, param.m_alpha, param.m_eta, param.m_stepSize);
        } else if (param.m_model.equals("multirole")) {
            model = new RoleEmbeddingBaseline(param.m_dim, param.m_nuOfRoles, param.m_iter, param.m_converge, param.m_alpha, param.m_beta, param.m_stepSize);
        } else if(param.m_model.equals("multirole_skipgram")){
            model = new RoleEmbeddingSkipGram(param.m_dim, param.m_nuOfRoles, param.m_iter, param.m_converge, param.m_alpha, param.m_beta, param.m_gamma, param.m_stepSize);
        } else {
            System.out.println("The model does not exist.....");
        }
        model.loadUsers(userFile);
        if(param.m_order >= 1)
            model.loadEdges(oneEdgeFile, 1); // load one edges
        if(param.m_order >= 2 )
            model.generate2ndConnections();
        if(param.m_order >= 3)
            model.generate3rdConnections();
        model.loadEdges(zeroEdgeFile, 0); // load zero edges

        model.train();
        if (param.m_model.equals("user")) {
            model.printUserEmbedding(baseEmbeddingFile);
        } else if (param.m_model.equals("user_skipgram")) {
            model.printUserEmbedding(userEmbeddingInputFile);
            ((UserEmbeddingSkipGram) model).printUserEmbeddingOutput(userEmbeddingOutputFile);
        } else if (param.m_model.equals("multirole")) {
            model.printUserEmbedding(userEmbeddingFile);
            ((RoleEmbeddingBaseline) model).printRoleEmbedding(roleEmbeddingFile, ((RoleEmbeddingBaseline) model).getRoleEmbeddings());
        } else if (param.m_model.equals("multirole_skipgram")){
            model.printUserEmbedding(userEmbeddingFile);
            ((RoleEmbeddingSkipGram) model).printRoleEmbedding(roleSourceEmbeddingFile, ((RoleEmbeddingSkipGram) model).getRoleEmbeddings());
            ((RoleEmbeddingSkipGram) model).printRoleEmbedding(roleTargetEmbeddingFile, ((RoleEmbeddingSkipGram) model).getContextRoleEmbeddings());
        }
    }
}