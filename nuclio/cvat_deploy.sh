#!/bin/bash

CVAT_REPO=$HOME/cvat
nuctl deploy --project-name cvat --path cvat_brixilated_maskrcnn --volume $CVAT_REPO/serverless/common:/opt/nuclio/common --platform local
