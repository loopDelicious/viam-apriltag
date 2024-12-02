## To run on Viam

Follow step-by-step instructions in this tutorial: 

### AprilTag detection

This module implements the [rdk vision API](https://docs.viam.com/appendix/apis/services/vision/) in a `joyce:vision:apriltag` model.

With this model, you can manage a vision service to detect and decode AprilTags.

### Build and Run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `joyce:vision:apriltag` model from the [`apriltag`](https://app.viam.com/module/joyce/apriltag) module.

### Configure your service

> [!NOTE]  
> Before configuring your sensor, you must [create a machine](https://docs.viam.com/cloud/machines/#add-a-new-machine).

- Navigate to the **CONFIGURE** tab of your robot’s page in [the Viam app](https://app.viam.com/).
- Click on the **+** icon in the left-hand menu and select **Service**.
- Select the `vision` type, then select the `apriltag` module. 
- Enter a name for your vision service and click **Create**.

> [!NOTE]  
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).
