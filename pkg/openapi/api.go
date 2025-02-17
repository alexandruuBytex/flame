// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Flame REST API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package openapi

import (
	"context"
	"net/http"
	"os"
)

// ComputesApiRouter defines the required methods for binding the api requests to a responses for the ComputesApi
// The ComputesApiRouter implementation should parse necessary information from the http request,
// pass the data to a ComputesApiServicer to perform the required actions, then write the service results to the http response.
type ComputesApiRouter interface {
	DeleteCompute(http.ResponseWriter, *http.Request)
	GetComputeConfig(http.ResponseWriter, *http.Request)
	GetComputeStatus(http.ResponseWriter, *http.Request)
	GetDeploymentConfig(http.ResponseWriter, *http.Request)
	GetDeploymentStatus(http.ResponseWriter, *http.Request)
	GetDeployments(http.ResponseWriter, *http.Request)
	PutDeploymentStatus(http.ResponseWriter, *http.Request)
	RegisterCompute(http.ResponseWriter, *http.Request)
	UpdateCompute(http.ResponseWriter, *http.Request)
}

// DatasetsApiRouter defines the required methods for binding the api requests to a responses for the DatasetsApi
// The DatasetsApiRouter implementation should parse necessary information from the http request,
// pass the data to a DatasetsApiServicer to perform the required actions, then write the service results to the http response.
type DatasetsApiRouter interface {
	CreateDataset(http.ResponseWriter, *http.Request)
	GetAllDatasets(http.ResponseWriter, *http.Request)
	GetDataset(http.ResponseWriter, *http.Request)
	GetDatasets(http.ResponseWriter, *http.Request)
	UpdateDataset(http.ResponseWriter, *http.Request)
}

// DesignCodesApiRouter defines the required methods for binding the api requests to a responses for the DesignCodesApi
// The DesignCodesApiRouter implementation should parse necessary information from the http request,
// pass the data to a DesignCodesApiServicer to perform the required actions, then write the service results to the http response.
type DesignCodesApiRouter interface {
	CreateDesignCode(http.ResponseWriter, *http.Request)
	DeleteDesignCode(http.ResponseWriter, *http.Request)
	GetDesignCode(http.ResponseWriter, *http.Request)
	UpdateDesignCode(http.ResponseWriter, *http.Request)
}

// DesignSchemasApiRouter defines the required methods for binding the api requests to a responses for the DesignSchemasApi
// The DesignSchemasApiRouter implementation should parse necessary information from the http request,
// pass the data to a DesignSchemasApiServicer to perform the required actions, then write the service results to the http response.
type DesignSchemasApiRouter interface {
	CreateDesignSchema(http.ResponseWriter, *http.Request)
	DeleteDesignSchema(http.ResponseWriter, *http.Request)
	GetDesignSchema(http.ResponseWriter, *http.Request)
	GetDesignSchemas(http.ResponseWriter, *http.Request)
	UpdateDesignSchema(http.ResponseWriter, *http.Request)
}

// DesignsApiRouter defines the required methods for binding the api requests to a responses for the DesignsApi
// The DesignsApiRouter implementation should parse necessary information from the http request,
// pass the data to a DesignsApiServicer to perform the required actions, then write the service results to the http response.
type DesignsApiRouter interface {
	CreateDesign(http.ResponseWriter, *http.Request)
	DeleteDesign(http.ResponseWriter, *http.Request)
	GetDesign(http.ResponseWriter, *http.Request)
	GetDesigns(http.ResponseWriter, *http.Request)
}

// JobsApiRouter defines the required methods for binding the api requests to a responses for the JobsApi
// The JobsApiRouter implementation should parse necessary information from the http request,
// pass the data to a JobsApiServicer to perform the required actions, then write the service results to the http response.
type JobsApiRouter interface {
	CreateJob(http.ResponseWriter, *http.Request)
	DeleteJob(http.ResponseWriter, *http.Request)
	GetJob(http.ResponseWriter, *http.Request)
	GetJobStatus(http.ResponseWriter, *http.Request)
	GetJobs(http.ResponseWriter, *http.Request)
	GetTask(http.ResponseWriter, *http.Request)
	GetTaskInfo(http.ResponseWriter, *http.Request)
	GetTasksInfo(http.ResponseWriter, *http.Request)
	UpdateJob(http.ResponseWriter, *http.Request)
	UpdateJobStatus(http.ResponseWriter, *http.Request)
	UpdateTaskStatus(http.ResponseWriter, *http.Request)
}

// ComputesApiServicer defines the api actions for the ComputesApi service
// This interface intended to stay up to date with the openapi yaml used to generate it,
// while the service implementation can be ignored with the .openapi-generator-ignore file
// and updated with the logic required for the API.
type ComputesApiServicer interface {
	DeleteCompute(context.Context, string, string) (ImplResponse, error)
	GetComputeConfig(context.Context, string, string) (ImplResponse, error)
	GetComputeStatus(context.Context, string, string) (ImplResponse, error)
	GetDeploymentConfig(context.Context, string, string, string) (ImplResponse, error)
	GetDeploymentStatus(context.Context, string, string, string) (ImplResponse, error)
	GetDeployments(context.Context, string, string) (ImplResponse, error)
	PutDeploymentStatus(context.Context, string, string, string, map[string]AgentState) (ImplResponse, error)
	RegisterCompute(context.Context, ComputeSpec) (ImplResponse, error)
	UpdateCompute(context.Context, string, string, ComputeSpec) (ImplResponse, error)
}

// DatasetsApiServicer defines the api actions for the DatasetsApi service
// This interface intended to stay up to date with the openapi yaml used to generate it,
// while the service implementation can be ignored with the .openapi-generator-ignore file
// and updated with the logic required for the API.
type DatasetsApiServicer interface {
	CreateDataset(context.Context, string, DatasetInfo) (ImplResponse, error)
	GetAllDatasets(context.Context, int32) (ImplResponse, error)
	GetDataset(context.Context, string, string) (ImplResponse, error)
	GetDatasets(context.Context, string, int32) (ImplResponse, error)
	UpdateDataset(context.Context, string, string, DatasetInfo) (ImplResponse, error)
}

// DesignCodesApiServicer defines the api actions for the DesignCodesApi service
// This interface intended to stay up to date with the openapi yaml used to generate it,
// while the service implementation can be ignored with the .openapi-generator-ignore file
// and updated with the logic required for the API.
type DesignCodesApiServicer interface {
	CreateDesignCode(context.Context, string, string, string, string, *os.File) (ImplResponse, error)
	DeleteDesignCode(context.Context, string, string, string) (ImplResponse, error)
	GetDesignCode(context.Context, string, string, string) (ImplResponse, error)
	UpdateDesignCode(context.Context, string, string, string, string, string, *os.File) (ImplResponse, error)
}

// DesignSchemasApiServicer defines the api actions for the DesignSchemasApi service
// This interface intended to stay up to date with the openapi yaml used to generate it,
// while the service implementation can be ignored with the .openapi-generator-ignore file
// and updated with the logic required for the API.
type DesignSchemasApiServicer interface {
	CreateDesignSchema(context.Context, string, string, DesignSchema) (ImplResponse, error)
	DeleteDesignSchema(context.Context, string, string, string) (ImplResponse, error)
	GetDesignSchema(context.Context, string, string, string) (ImplResponse, error)
	GetDesignSchemas(context.Context, string, string) (ImplResponse, error)
	UpdateDesignSchema(context.Context, string, string, string, DesignSchema) (ImplResponse, error)
}

// DesignsApiServicer defines the api actions for the DesignsApi service
// This interface intended to stay up to date with the openapi yaml used to generate it,
// while the service implementation can be ignored with the .openapi-generator-ignore file
// and updated with the logic required for the API.
type DesignsApiServicer interface {
	CreateDesign(context.Context, string, DesignInfo) (ImplResponse, error)
	DeleteDesign(context.Context, string, string) (ImplResponse, error)
	GetDesign(context.Context, string, string) (ImplResponse, error)
	GetDesigns(context.Context, string, int32) (ImplResponse, error)
}

// JobsApiServicer defines the api actions for the JobsApi service
// This interface intended to stay up to date with the openapi yaml used to generate it,
// while the service implementation can be ignored with the .openapi-generator-ignore file
// and updated with the logic required for the API.
type JobsApiServicer interface {
	CreateJob(context.Context, string, JobSpec) (ImplResponse, error)
	DeleteJob(context.Context, string, string) (ImplResponse, error)
	GetJob(context.Context, string, string) (ImplResponse, error)
	GetJobStatus(context.Context, string, string) (ImplResponse, error)
	GetJobs(context.Context, string, int32) (ImplResponse, error)
	GetTask(context.Context, string, string, string) (ImplResponse, error)
	GetTaskInfo(context.Context, string, string, string) (ImplResponse, error)
	GetTasksInfo(context.Context, string, string, int32) (ImplResponse, error)
	UpdateJob(context.Context, string, string, JobSpec) (ImplResponse, error)
	UpdateJobStatus(context.Context, string, string, JobStatus) (ImplResponse, error)
	UpdateTaskStatus(context.Context, string, string, TaskStatus) (ImplResponse, error)
}
