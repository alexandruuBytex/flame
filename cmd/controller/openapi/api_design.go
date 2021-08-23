/*
 * Fledge REST API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package openapi

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/gorilla/mux"
	objects2 "wwwin-github.cisco.com/eti/fledge/pkg/objects"
)

// A DesignApiController binds http requests to an api service and writes the service results to the http response
type DesignApiController struct {
	service DesignApiServicer
}

// NewDesignApiController creates a default api controller
func NewDesignApiController(s DesignApiServicer) Router {
	return &DesignApiController{service: s}
}

// Routes returns all of the api route for the DesignApiController
func (c *DesignApiController) Routes() Routes {
	return Routes{
		{
			"CreateDesign",
			strings.ToUpper("Post"),
			"/{user}/design/",
			c.CreateDesign,
		},
		{
			"GetDesign",
			strings.ToUpper("Get"),
			"/{user}/design/{designId}",
			c.GetDesign,
		},
	}
}

// CreateDesign - Create a new design template.
func (c *DesignApiController) CreateDesign(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	user := params["user"]

	designInfo := &objects2.DesignInfo{}
	if err := json.NewDecoder(r.Body).Decode(&designInfo); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	result, err := c.service.CreateDesign(r.Context(), user, *designInfo)
	// If an error occurred, encode the error with the status code
	if err != nil {
		EncodeJSONResponse(err.Error(), &result.Code, w)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}

// GetDesign - Get design template information
func (c *DesignApiController) GetDesign(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	user := params["user"]

	designId := params["designId"]

	result, err := c.service.GetDesign(r.Context(), user, designId)
	// If an error occurred, encode the error with the status code
	if err != nil {
		EncodeJSONResponse(err.Error(), &result.Code, w)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}
