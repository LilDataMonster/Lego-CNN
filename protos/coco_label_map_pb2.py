# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: coco_label_map.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='coco_label_map.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_pb=_b('\n\x14\x63oco_label_map.proto\x12\x17object_detection.protos\"\xc1\x02\n\x15StringIntLabelMapItem\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\x05\x12\x14\n\x0c\x64isplay_name\x18\x03 \x01(\t\x12M\n\tkeypoints\x18\x04 \x03(\x0b\x32:.object_detection.protos.StringIntLabelMapItem.KeypointMap\x12\x14\n\x0c\x61ncestor_ids\x18\x05 \x03(\x05\x12\x16\n\x0e\x64\x65scendant_ids\x18\x06 \x03(\x05\x12\x39\n\tfrequency\x18\x07 \x01(\x0e\x32&.object_detection.protos.LVISFrequency\x12\x16\n\x0einstance_count\x18\x08 \x01(\x05\x1a(\n\x0bKeypointMap\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05label\x18\x02 \x01(\t\"Q\n\x11StringIntLabelMap\x12<\n\x04item\x18\x01 \x03(\x0b\x32..object_detection.protos.StringIntLabelMapItem*D\n\rLVISFrequency\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\x0c\n\x08\x46REQUENT\x10\x01\x12\n\n\x06\x43OMMON\x10\x02\x12\x08\n\x04RARE\x10\x03')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_LVISFREQUENCY = _descriptor.EnumDescriptor(
  name='LVISFrequency',
  full_name='object_detection.protos.LVISFrequency',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNSPECIFIED', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FREQUENT', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COMMON', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RARE', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=456,
  serialized_end=524,
)
_sym_db.RegisterEnumDescriptor(_LVISFREQUENCY)

LVISFrequency = enum_type_wrapper.EnumTypeWrapper(_LVISFREQUENCY)
UNSPECIFIED = 0
FREQUENT = 1
COMMON = 2
RARE = 3



_STRINGINTLABELMAPITEM_KEYPOINTMAP = _descriptor.Descriptor(
  name='KeypointMap',
  full_name='object_detection.protos.StringIntLabelMapItem.KeypointMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='object_detection.protos.StringIntLabelMapItem.KeypointMap.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='label', full_name='object_detection.protos.StringIntLabelMapItem.KeypointMap.label', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=331,
  serialized_end=371,
)

_STRINGINTLABELMAPITEM = _descriptor.Descriptor(
  name='StringIntLabelMapItem',
  full_name='object_detection.protos.StringIntLabelMapItem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='object_detection.protos.StringIntLabelMapItem.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='id', full_name='object_detection.protos.StringIntLabelMapItem.id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='display_name', full_name='object_detection.protos.StringIntLabelMapItem.display_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='keypoints', full_name='object_detection.protos.StringIntLabelMapItem.keypoints', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ancestor_ids', full_name='object_detection.protos.StringIntLabelMapItem.ancestor_ids', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='descendant_ids', full_name='object_detection.protos.StringIntLabelMapItem.descendant_ids', index=5,
      number=6, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='frequency', full_name='object_detection.protos.StringIntLabelMapItem.frequency', index=6,
      number=7, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='instance_count', full_name='object_detection.protos.StringIntLabelMapItem.instance_count', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_STRINGINTLABELMAPITEM_KEYPOINTMAP, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=371,
)


_STRINGINTLABELMAP = _descriptor.Descriptor(
  name='StringIntLabelMap',
  full_name='object_detection.protos.StringIntLabelMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='item', full_name='object_detection.protos.StringIntLabelMap.item', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=373,
  serialized_end=454,
)

_STRINGINTLABELMAPITEM_KEYPOINTMAP.containing_type = _STRINGINTLABELMAPITEM
_STRINGINTLABELMAPITEM.fields_by_name['keypoints'].message_type = _STRINGINTLABELMAPITEM_KEYPOINTMAP
_STRINGINTLABELMAPITEM.fields_by_name['frequency'].enum_type = _LVISFREQUENCY
_STRINGINTLABELMAP.fields_by_name['item'].message_type = _STRINGINTLABELMAPITEM
DESCRIPTOR.message_types_by_name['StringIntLabelMapItem'] = _STRINGINTLABELMAPITEM
DESCRIPTOR.message_types_by_name['StringIntLabelMap'] = _STRINGINTLABELMAP
DESCRIPTOR.enum_types_by_name['LVISFrequency'] = _LVISFREQUENCY

StringIntLabelMapItem = _reflection.GeneratedProtocolMessageType('StringIntLabelMapItem', (_message.Message,), dict(

  KeypointMap = _reflection.GeneratedProtocolMessageType('KeypointMap', (_message.Message,), dict(
    DESCRIPTOR = _STRINGINTLABELMAPITEM_KEYPOINTMAP,
    __module__ = 'coco_label_map_pb2'
    # @@protoc_insertion_point(class_scope:object_detection.protos.StringIntLabelMapItem.KeypointMap)
    ))
  ,
  DESCRIPTOR = _STRINGINTLABELMAPITEM,
  __module__ = 'coco_label_map_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.StringIntLabelMapItem)
  ))
_sym_db.RegisterMessage(StringIntLabelMapItem)
_sym_db.RegisterMessage(StringIntLabelMapItem.KeypointMap)

StringIntLabelMap = _reflection.GeneratedProtocolMessageType('StringIntLabelMap', (_message.Message,), dict(
  DESCRIPTOR = _STRINGINTLABELMAP,
  __module__ = 'coco_label_map_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.StringIntLabelMap)
  ))
_sym_db.RegisterMessage(StringIntLabelMap)


# @@protoc_insertion_point(module_scope)
