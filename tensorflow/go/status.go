/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

// #include "tensorflow/c/c_api.h"
import "C"

import "runtime"

type code C.TF_Code

// status holds error information returned by TensorFlow. We convert all
// TF statuses to Go errors (or nil if the status is OK)
type status struct {
	c *C.TF_Status
}

func newStatus() *status {
	s := &status{C.TF_NewStatus()}
	runtime.SetFinalizer(s, (*status).finalizer)
	return s
}

// This finalizer is belt-and-braces and ideally would be removed. It should not
// be needed as creators of status should always call close
func (s *status) finalizer() {
	C.TF_DeleteStatus(s.c)
}

func (s *status) close() {
	if s.c != nil {
		runtime.SetFinalizer(s, nil)
		C.TF_DeleteStatus(s.c)
		s.c = nil
	}
}

func (s *status) code() code {
	if s.c == nil {
		return C.TF_OK
	}
	return code(C.TF_GetCode(s.c))
}

func (s *status) String() string {
	return C.GoString(C.TF_Message(s.c))
}

// Err converts the status to a Go error and returns nil if the status is OK.
// Note it does not need to be public!
func (s *status) Err() error {
	if s == nil || s.code() == C.TF_OK {
		return nil
	}
	// Note that we don't want to rely on the C memory past this point. If we
	// want to present the code to the caller we should wrap it into the error
	// at this juncture.
	return &statusError{msg: s.String()}
}

// statusError is distinct from status because it fulfills the error interface.
// status itself may have a TF_OK code and is not always considered an error.
//
// TODO(jhseu): Make public, rename to Error, and provide a way for users to
// check status codes.
type statusError struct {
	msg string
}

func (s *statusError) Error() string {
	return s.msg
}
