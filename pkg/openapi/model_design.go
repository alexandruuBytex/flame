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

// Design - Design template details along with all the schemas.
type Design struct {
	Name string `json:"name"`

	Description string `json:"description,omitempty"`

	Id string `json:"id"`

	UserId string `json:"userId,omitempty"`

	Schemas []DesignSchema `json:"schemas"`
}

// AssertDesignRequired checks if the required fields are not zero-ed
func AssertDesignRequired(obj Design) error {
	elements := map[string]interface{}{
<<<<<<< HEAD
=======
		"name":    obj.Name,
>>>>>>> cc8e286 (Sync up the generated code from openapi generator with what we have currently (#331))
		"id":      obj.Id,
		"schemas": obj.Schemas,
	}
	for name, el := range elements {
		if isZero := IsZeroValue(el); isZero {
			return &RequiredError{Field: name}
		}
	}

	for _, el := range obj.Schemas {
		if err := AssertDesignSchemaRequired(el); err != nil {
			return err
		}
	}
	return nil
}

// AssertRecurseDesignRequired recursively checks if required fields are not zero-ed in a nested slice.
// Accepts only nested slice of Design (e.g. [][]Design), otherwise ErrTypeAssertionError is thrown.
func AssertRecurseDesignRequired(objSlice interface{}) error {
	return AssertRecurseInterfaceRequired(objSlice, func(obj interface{}) error {
		aDesign, ok := obj.(Design)
		if !ok {
			return ErrTypeAssertionError
		}
		return AssertDesignRequired(aDesign)
	})
}
