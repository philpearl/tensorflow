// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//==============================================================================

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.21.0-devel
// 	protoc        v3.10.1
// source: tensorflow/core/protobuf/device_filters.proto

package core_protos_go_proto

import (
	proto "github.com/golang/protobuf/proto"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// This is a compile-time assertion that a sufficiently up-to-date version
// of the legacy proto package is being used.
const _ = proto.ProtoPackageIsVersion4

// Defines the device filters for a remote task.
type TaskDeviceFilters struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	DeviceFilters []string `protobuf:"bytes,1,rep,name=device_filters,json=deviceFilters,proto3" json:"device_filters,omitempty"`
}

func (x *TaskDeviceFilters) Reset() {
	*x = TaskDeviceFilters{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensorflow_core_protobuf_device_filters_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *TaskDeviceFilters) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TaskDeviceFilters) ProtoMessage() {}

func (x *TaskDeviceFilters) ProtoReflect() protoreflect.Message {
	mi := &file_tensorflow_core_protobuf_device_filters_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use TaskDeviceFilters.ProtoReflect.Descriptor instead.
func (*TaskDeviceFilters) Descriptor() ([]byte, []int) {
	return file_tensorflow_core_protobuf_device_filters_proto_rawDescGZIP(), []int{0}
}

func (x *TaskDeviceFilters) GetDeviceFilters() []string {
	if x != nil {
		return x.DeviceFilters
	}
	return nil
}

// Defines the device filters for tasks in a job.
type JobDeviceFilters struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The name of this job.
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	// Mapping from task ID to task device filters.
	Tasks map[int32]*TaskDeviceFilters `protobuf:"bytes,2,rep,name=tasks,proto3" json:"tasks,omitempty" protobuf_key:"varint,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
}

func (x *JobDeviceFilters) Reset() {
	*x = JobDeviceFilters{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensorflow_core_protobuf_device_filters_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *JobDeviceFilters) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*JobDeviceFilters) ProtoMessage() {}

func (x *JobDeviceFilters) ProtoReflect() protoreflect.Message {
	mi := &file_tensorflow_core_protobuf_device_filters_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use JobDeviceFilters.ProtoReflect.Descriptor instead.
func (*JobDeviceFilters) Descriptor() ([]byte, []int) {
	return file_tensorflow_core_protobuf_device_filters_proto_rawDescGZIP(), []int{1}
}

func (x *JobDeviceFilters) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *JobDeviceFilters) GetTasks() map[int32]*TaskDeviceFilters {
	if x != nil {
		return x.Tasks
	}
	return nil
}

// Defines the device filters for jobs in a cluster.
type ClusterDeviceFilters struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Jobs []*JobDeviceFilters `protobuf:"bytes,1,rep,name=jobs,proto3" json:"jobs,omitempty"`
}

func (x *ClusterDeviceFilters) Reset() {
	*x = ClusterDeviceFilters{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensorflow_core_protobuf_device_filters_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ClusterDeviceFilters) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ClusterDeviceFilters) ProtoMessage() {}

func (x *ClusterDeviceFilters) ProtoReflect() protoreflect.Message {
	mi := &file_tensorflow_core_protobuf_device_filters_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ClusterDeviceFilters.ProtoReflect.Descriptor instead.
func (*ClusterDeviceFilters) Descriptor() ([]byte, []int) {
	return file_tensorflow_core_protobuf_device_filters_proto_rawDescGZIP(), []int{2}
}

func (x *ClusterDeviceFilters) GetJobs() []*JobDeviceFilters {
	if x != nil {
		return x.Jobs
	}
	return nil
}

var File_tensorflow_core_protobuf_device_filters_proto protoreflect.FileDescriptor

var file_tensorflow_core_protobuf_device_filters_proto_rawDesc = []byte{
	0x0a, 0x2d, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f, 0x63, 0x6f, 0x72,
	0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x64, 0x65, 0x76, 0x69, 0x63,
	0x65, 0x5f, 0x66, 0x69, 0x6c, 0x74, 0x65, 0x72, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12,
	0x0a, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x22, 0x3a, 0x0a, 0x11, 0x54,
	0x61, 0x73, 0x6b, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x46, 0x69, 0x6c, 0x74, 0x65, 0x72, 0x73,
	0x12, 0x25, 0x0a, 0x0e, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x5f, 0x66, 0x69, 0x6c, 0x74, 0x65,
	0x72, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x09, 0x52, 0x0d, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65,
	0x46, 0x69, 0x6c, 0x74, 0x65, 0x72, 0x73, 0x22, 0xbe, 0x01, 0x0a, 0x10, 0x4a, 0x6f, 0x62, 0x44,
	0x65, 0x76, 0x69, 0x63, 0x65, 0x46, 0x69, 0x6c, 0x74, 0x65, 0x72, 0x73, 0x12, 0x12, 0x0a, 0x04,
	0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65,
	0x12, 0x3d, 0x0a, 0x05, 0x74, 0x61, 0x73, 0x6b, 0x73, 0x18, 0x02, 0x20, 0x03, 0x28, 0x0b, 0x32,
	0x27, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2e, 0x4a, 0x6f, 0x62,
	0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x46, 0x69, 0x6c, 0x74, 0x65, 0x72, 0x73, 0x2e, 0x54, 0x61,
	0x73, 0x6b, 0x73, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x52, 0x05, 0x74, 0x61, 0x73, 0x6b, 0x73, 0x1a,
	0x57, 0x0a, 0x0a, 0x54, 0x61, 0x73, 0x6b, 0x73, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x12, 0x10, 0x0a,
	0x03, 0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x03, 0x6b, 0x65, 0x79, 0x12,
	0x33, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x1d,
	0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2e, 0x54, 0x61, 0x73, 0x6b,
	0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x46, 0x69, 0x6c, 0x74, 0x65, 0x72, 0x73, 0x52, 0x05, 0x76,
	0x61, 0x6c, 0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x22, 0x48, 0x0a, 0x14, 0x43, 0x6c, 0x75, 0x73,
	0x74, 0x65, 0x72, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x46, 0x69, 0x6c, 0x74, 0x65, 0x72, 0x73,
	0x12, 0x30, 0x0a, 0x04, 0x6a, 0x6f, 0x62, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x1c,
	0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2e, 0x4a, 0x6f, 0x62, 0x44,
	0x65, 0x76, 0x69, 0x63, 0x65, 0x46, 0x69, 0x6c, 0x74, 0x65, 0x72, 0x73, 0x52, 0x04, 0x6a, 0x6f,
	0x62, 0x73, 0x42, 0x80, 0x01, 0x0a, 0x1a, 0x6f, 0x72, 0x67, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f,
	0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2e, 0x64, 0x69, 0x73, 0x74, 0x72, 0x75, 0x6e, 0x74, 0x69, 0x6d,
	0x65, 0x42, 0x13, 0x44, 0x65, 0x76, 0x69, 0x63, 0x65, 0x46, 0x69, 0x6c, 0x74, 0x65, 0x72, 0x73,
	0x50, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x50, 0x01, 0x5a, 0x48, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62,
	0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f,
	0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f, 0x74, 0x65, 0x6e, 0x73, 0x6f,
	0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f, 0x67, 0x6f, 0x2f, 0x63, 0x6f, 0x72, 0x65, 0x2f, 0x63, 0x6f,
	0x72, 0x65, 0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x5f, 0x67, 0x6f, 0x5f, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0xf8, 0x01, 0x01, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_tensorflow_core_protobuf_device_filters_proto_rawDescOnce sync.Once
	file_tensorflow_core_protobuf_device_filters_proto_rawDescData = file_tensorflow_core_protobuf_device_filters_proto_rawDesc
)

func file_tensorflow_core_protobuf_device_filters_proto_rawDescGZIP() []byte {
	file_tensorflow_core_protobuf_device_filters_proto_rawDescOnce.Do(func() {
		file_tensorflow_core_protobuf_device_filters_proto_rawDescData = protoimpl.X.CompressGZIP(file_tensorflow_core_protobuf_device_filters_proto_rawDescData)
	})
	return file_tensorflow_core_protobuf_device_filters_proto_rawDescData
}

var file_tensorflow_core_protobuf_device_filters_proto_msgTypes = make([]protoimpl.MessageInfo, 4)
var file_tensorflow_core_protobuf_device_filters_proto_goTypes = []interface{}{
	(*TaskDeviceFilters)(nil),    // 0: tensorflow.TaskDeviceFilters
	(*JobDeviceFilters)(nil),     // 1: tensorflow.JobDeviceFilters
	(*ClusterDeviceFilters)(nil), // 2: tensorflow.ClusterDeviceFilters
	nil,                          // 3: tensorflow.JobDeviceFilters.TasksEntry
}
var file_tensorflow_core_protobuf_device_filters_proto_depIdxs = []int32{
	3, // 0: tensorflow.JobDeviceFilters.tasks:type_name -> tensorflow.JobDeviceFilters.TasksEntry
	1, // 1: tensorflow.ClusterDeviceFilters.jobs:type_name -> tensorflow.JobDeviceFilters
	0, // 2: tensorflow.JobDeviceFilters.TasksEntry.value:type_name -> tensorflow.TaskDeviceFilters
	3, // [3:3] is the sub-list for method output_type
	3, // [3:3] is the sub-list for method input_type
	3, // [3:3] is the sub-list for extension type_name
	3, // [3:3] is the sub-list for extension extendee
	0, // [0:3] is the sub-list for field type_name
}

func init() { file_tensorflow_core_protobuf_device_filters_proto_init() }
func file_tensorflow_core_protobuf_device_filters_proto_init() {
	if File_tensorflow_core_protobuf_device_filters_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_tensorflow_core_protobuf_device_filters_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*TaskDeviceFilters); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_tensorflow_core_protobuf_device_filters_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*JobDeviceFilters); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_tensorflow_core_protobuf_device_filters_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ClusterDeviceFilters); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_tensorflow_core_protobuf_device_filters_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   4,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_tensorflow_core_protobuf_device_filters_proto_goTypes,
		DependencyIndexes: file_tensorflow_core_protobuf_device_filters_proto_depIdxs,
		MessageInfos:      file_tensorflow_core_protobuf_device_filters_proto_msgTypes,
	}.Build()
	File_tensorflow_core_protobuf_device_filters_proto = out.File
	file_tensorflow_core_protobuf_device_filters_proto_rawDesc = nil
	file_tensorflow_core_protobuf_device_filters_proto_goTypes = nil
	file_tensorflow_core_protobuf_device_filters_proto_depIdxs = nil
}
