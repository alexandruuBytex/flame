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

import "github.com/cisco-open/flame/pkg/openapi/constants"

// DesignSchema - Schema to define the roles and their connections
type DesignSchema struct {
	Version string `json:"version"`

	Name string `json:"name"`

	Description string `json:"description,omitempty"`

	Roles []Role `json:"roles"`

	Channels []Channel `json:"channels"`

	Connectors []Connector `json:"connectors,omitempty"`
}

// AssertDesignSchemaRequired checks if the required fields are not zero-ed
func AssertDesignSchemaRequired(obj DesignSchema) error {
	elements := map[string]interface{}{
<<<<<<< HEAD
		"name":     obj.Name,
		"roles":    obj.Roles,
		"channels": obj.Channels,
=======
		constants.ParamVersion: obj.Version,
		"name":                 obj.Name,
		"roles":                obj.Roles,
		"channels":             obj.Channels,
>>>>>>> cc8e286 (Sync up the generated code from openapi generator with what we have currently (#331))
	}
	for name, el := range elements {
		if isZero := IsZeroValue(el); isZero {
			return &RequiredError{Field: name}
		}
	}

	for _, el := range obj.Roles {
		if err := AssertRoleRequired(el); err != nil {
			return err
		}
	}
	for _, el := range obj.Channels {
		if err := AssertChannelRequired(el); err != nil {
			return err
		}
	}
	for _, el := range obj.Connectors {
		if err := AssertConnectorRequired(el); err != nil {
			return err
		}
	}
	return nil
}

// AssertRecurseDesignSchemaRequired recursively checks if required fields are not zero-ed in a nested slice.
// Accepts only nested slice of DesignSchema (e.g. [][]DesignSchema), otherwise ErrTypeAssertionError is thrown.
func AssertRecurseDesignSchemaRequired(objSlice interface{}) error {
	return AssertRecurseInterfaceRequired(objSlice, func(obj interface{}) error {
		aDesignSchema, ok := obj.(DesignSchema)
		if !ok {
			return ErrTypeAssertionError
		}
		return AssertDesignSchemaRequired(aDesignSchema)
	})
}
