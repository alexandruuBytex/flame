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
	// "encoding/json"
	"net/http"
	"strings"

	"github.com/gorilla/mux"
)

// A DesignsApiController binds http requests to an api service and writes the service results to the http response
type DesignsApiController struct {
	service DesignsApiServicer
}

// NewDesignsApiController creates a default api controller
func NewDesignsApiController(s DesignsApiServicer) Router {
	return &DesignsApiController{service: s}
}

// Routes returns all of the api route for the DesignsApiController
func (c *DesignsApiController) Routes() Routes {
	return Routes{
		{
			"CreateDesigns",
			strings.ToUpper("Post"),
			"/designs/{user}",
			c.CreateDesigns,
		},
		{
			"ListDesigns",
			strings.ToUpper("Get"),
			"/designs/{user}",
			c.ListDesigns,
		},
	}
}

// CreateDesigns - Create a design
func (c *DesignsApiController) CreateDesigns(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	query := r.URL.Query()
	user := params["user"]

	name := query.Get("name")
	result, err := c.service.CreateDesigns(r.Context(), user, name)
	// If an error occurred, encode the error with the status code
	if err != nil {
		EncodeJSONResponse(err.Error(), &result.Code, w)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)

}

// ListDesigns - List designs owned by user
func (c *DesignsApiController) ListDesigns(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	query := r.URL.Query()
	user := params["user"]

	limit, err := parseInt32Parameter(query.Get("limit"), false)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	result, err := c.service.ListDesigns(r.Context(), user, limit)
	// If an error occurred, encode the error with the status code
	if err != nil {
		EncodeJSONResponse(err.Error(), &result.Code, w)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)

}
