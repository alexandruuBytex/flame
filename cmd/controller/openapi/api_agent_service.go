/*
 * Job REST API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package openapi

import (
	"context"
	"errors"
	"net/http"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/database"
	grpcctlr "wwwin-github.cisco.com/eti/fledge/cmd/controller/grpc"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

// AgentApiService is a service that implents the logic for the AgentApiServicer
// This service should implement the business logic for every endpoint for the AgentApi API.
// Include any external packages or services that will be required by this service.
type AgentApiService struct {
}

// NewAgentApiService creates a default api service
func NewAgentApiService() AgentApiServicer {
	return &AgentApiService{}
}

// UpdateAgentStatus - Update agent status for job id.
func (s *AgentApiService) UpdateAgentStatus(ctx context.Context, user string, jobId string, agentId string,
	agentStatus objects.AgentStatus) (objects.ImplResponse, error) {
	zap.S().Debugf("Update agent status agentId: %s | jobId: %s | update type: %s", agentId, jobId, agentStatus.UpdateType)

	// JobStatus implies agent sending a status about the init/start state of the application.
	// When a job is submitted the first, notification from the agents for a non data consumer nodes
	// should be running/error while for a data consumer node it would be ready.
	// As applications start to come up, after every update we will check if the trainers
	// (also called as data consumers) are required to be started.
	if agentStatus.UpdateType == util.JobStatus {
		//Step 1 - status update
		err := updateJobStatus(jobId, agentId, agentStatus)
		if err != nil {
			return objects.Response(http.StatusInternalServerError, err), err
		}

		//Step 2 - check and start training process
		err = startTraining(jobId, agentId)
		if err != nil {
			return objects.Response(http.StatusInternalServerError, err), err
		}
	} else {
		zap.S().Errorf("Invalid update type.")
		return objects.Response(http.StatusInternalServerError, nil), errors.New("invalid update type")
	}

	return objects.Response(http.StatusOK, nil), nil
}

//startTraining first validate the starting policy and sends notification to the trainer (aka data consumer nodes)
func startTraining(jobId string, agentId string) error {
	//implementing a simple policy where once all the non training nodes are up send the job start notification to the training nodes.
	zap.S().Debugf("Checking job startup policy.")

	//determine the trainer role
	dataConsumerRole := ""
	for _, role := range Cache.jobSchema[jobId].Roles {
		if role.IsDataConsumer {
			dataConsumerRole = role.Name
			break
		}
	}

	//policy check - simple policy of verifying all the non-DC nodes are up and running.
	var dcAgentList []objects.ServerInfo
	sendStartSignal := true
	for _, agent := range Cache.jobAgents[jobId] {
		//check if it is a DC node and is not running
		if agent.Role == dataConsumerRole && agent.State != util.RunningState {
			dcAgentList = append(dcAgentList, agent)
		} else if agent.State != util.RunningState {
			//TODO add a group by rule to ensure we are checking only in the corresponding group of nodes.
			//For example - when checking for 'us' group - only the nodes with 'us' tags are checked and 'uk' are ignored.
			zap.S().Debugf("Breaking off becaues one of the agent (non dc) role: %s | state: %s | uuid: %s ", agent.Role, agent.State, agent.Uuid)
			sendStartSignal = false
			break
		}
	}

	//sending notification
	if sendStartSignal && len(dcAgentList) > 0 {
		//notify all
		jobMsg := objects.JobNotification{
			Agents: dcAgentList,
			Job: objects.JobInfo{
				ID: jobId,
			},
			SchemaInfo:       objects.DesignSchema{},
			NotificationType: util.StartState,
		}
		zap.S().Debugf("Sending start job notification to the trainer agents (count: %d) for job id: %s", len(dcAgentList), jobId)
		resp, err := grpcctlr.ControllerGRPC.SendNotification(grpcctlr.JobNotification, jobMsg)
		if err != nil {
			zap.S().Errorf("failed to notify the agents. %v", err)
			return errors.New("error while sending job start notification")
		}

		//Check for partial error
		if resp.GetStatus() == pbNotification.Response_SUCCESS_WITH_ERROR {
			zap.S().Errorf("error while sending out start job notification for jobId: %s. Only partial agents were notified.", jobId)
			return errors.New("error while sending job start notification to some agents")
		}
	} else {
		zap.S().Debugf("Not sending start job notification now for job id:%s. Check trigged by uuid: %s", jobId, agentId)
	}

	return nil
}

func updateJobStatus(jobId string, agentId string, agentStatus objects.AgentStatus) error {
	isFound := false
	for index, agent := range Cache.jobAgents[jobId] {
		if agent.Uuid == agentId {
			state := util.ErrorState
			if agentStatus.Status == util.StatusSuccess {
				state = agentStatus.Message
			}
			//update database
			err := database.UpdateJobDetails(jobId, util.JobStatus, map[string]string{
				util.ID:    agentId,
				util.State: state,
			})
			if err != nil {
				zap.S().Errorf("error updating the node status in db")
				return err
			}

			//update cache
			Cache.jobAgents[jobId][index].State = state
			isFound = true
			break //because you have found the agent node and can avoid iterating the other nodes.
		}
	}

	if !isFound {
		zap.S().Errorf("Not able to find the agent id %s for the given jobId %s", agentId, jobId)
		return errors.New("not able to find the agent for the given job")
	}
	return nil
}
